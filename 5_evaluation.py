#%%
import os
import sys
import time
import json
import pandas as pd
import pprint
import re
from tqdm import tqdm
import numpy as np

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from module.data_pipeline.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
#%%
data_dir = "./assets/data"
with open(f"{data_dir}/default_work.json", "r", encoding="utf-8") as f:
    default_work = json.load(f)
#%%
def evaluation(
    load_8bit: bool = True,
    base_model: str = "beomi/KoAlpaca-Polyglot-5.8B",
    lora_weights: str = "./models/korani_LORA_pretrained", # pretrained
):
    #%%
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    #%%
    """load model"""
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device}, 
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    #%%
    """tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    #%%
    """prompter"""
    prompter = Prompter("koalpaca")
    prompter_uos = Prompter("korani")
    #%%
    def evaluate(
        instruction,
        uos=True,
        batch_size=3, # for multiple answers
        input=None,
        temperature=0.5,
        topk=5,
        max_new_tokens=512,
        # stream_output=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ):
        if uos:
            torch.manual_seed(2) # fixed seed
            
            prompt = prompter_uos.generate_prompt(instruction, input)
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
            sequence = inputs['input_ids'].to('cuda:0')
            
            """top-K sampling"""
            # idx is (B, T) array of indices in the current context
            for i in range(max_new_tokens):
                # get the predictions
                with torch.no_grad():
                    model.float()
                    output = model.base_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        use_cache=False
                    )
                    
                # focus only on the last time step
                logits = output.logits[:, -1, :] # becomes (B, C)    
                # Only keep top-1 + top-K indices
                topk_logits = torch.cat(
                    [torch.topk(l, k)[0][[-1]].unsqueeze(0) 
                    for l, k in zip(logits, [1] + [topk] * (batch_size-1))], dim=0)
                indices_to_remove = logits < topk_logits
                logits[indices_to_remove] = torch.tensor(float('-inf')).to('cuda:0')
                # Convert logits to probabilities
                probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1).to('cuda:0')
                # Sample n tokens from the resulting distribution
                idx_next = torch.multinomial(probabilities, num_samples=1).to('cuda:0') # (B, 1)
                # append sampled index to the running sequence
                sequence = torch.cat((sequence, idx_next), dim=1) # (B, T+1)
                
                # get updated inputs
                next_inputs = [tokenizer.decode(s) for s in sequence]
                inputs = tokenizer(next_inputs, return_tensors="pt")
                
                if sequence[0, -1] == eos_token_id:
                    break

            output = [tokenizer.decode(s) for s in sequence]
            output = [prompter_uos.get_response(out) for out in output]
            output = [out.split(tokenizer.eos_token)[0] for out in output]

            outputs = []
            for out in output:
                """add manual response"""
                for key, value in default_work.items():
                    if key in out:
                        out += '\n' + value
                        break
                outputs.append(out)
            outputs = list(set(outputs))
                
            if len(outputs) > 1:
                result = ""
                for i, output in enumerate(outputs):
                    result += f"[답변 {i+1}]\n"
                    result += output + "\n\n"
            else:
                result = outputs[0]
            
        else: # hard-coded
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                do_sample=True, #####
                temperature=0.7,
                top_p=0.8,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            # Without streaming
            with torch.no_grad():
                model.float()
                generation_output = model.generate(
                    do_sample=True, #####
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            
            output = tokenizer.decode(generation_output.sequences[0])
            output = prompter.get_response(output)
            """remove end token"""
            if tokenizer._eos_token in output:
                result = output.replace(tokenizer._eos_token, "")
        return result
    #%%
    """Cherry picking"""
    for instruction in [
        "R&D기반조성사업 관련 문의는 누구에게 해야 하나요?",
        "중앙구매 담당자는 누구인가요?",
        "연구 윤리 관련 담당자 알려주세요.",
        "공과대 소속 연구자인데, 연구 과제 계획서 제출관련은 누구 담당이야?",
        "도과대 연구자입니다. 연구 지원금 담당자는 누구인가요?"
    ]:
        pred = evaluate(instruction)
        print('instruction:', instruction)
        print()
        print(pred)
        print()
    #%%
    """load test data"""
    data_dir = "./assets/data"
    test_df = pd.read_csv(f"{data_dir}/testset_v1.csv", encoding='utf-8')
    #%%
    score = []
    each_score = np.zeros((5, ))
    each_len = test_df.groupby('type').count()['output'].to_list()
    for i in tqdm(range(len(test_df))):
        target = test_df['output'].iloc[i]
        instruction = test_df['instruction'].iloc[i]
        
        start = time.time()
        pred = evaluate(instruction)
        end = time.time()
        
        answer = float(any([name in pred for name in target.split(", ")]))
        
        type_ = test_df['type'].iloc[i]
        each_score[type_] = each_score[type_] + answer
        
        score.append((
            instruction,
            target, 
            pred, 
            end - start,
            answer))
    #%%
    np.save(
        f"./assets/etc/topk_{lora_weights.split('/')[-1]}_inference_time", 
        np.array([x[-2] for x in score]))
    #%%
    with open(lora_weights + "/train_config.json", "r") as f:
        train_config = json.load(f)
    #%%
    for x in score:
        if x[-1] == 0:
            print('Instruction:', x[0])
            print('Target:', x[1])
            for name in x[1].split(", "):
                print(default_work.get(name))
            print('Pred:', x[2])
            print()
    
    each_score_df = pd.DataFrame([x / l * 100 for x, l in zip(each_score, each_len)])
    print(each_score_df)
    print()
    print(f"Accuracy: {np.mean([x[-1] for x in score])*100:.2f}%")
    print(f"Inference time: {np.mean([x[-2] for x in score]):.2f} sec/query")
    print()
    pprint.pprint(train_config)
#%%
if __name__ == '__main__':
    fire.Fire(evaluation)
#%%