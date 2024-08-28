#%%
import os
import sys
import json
import pandas as pd

import gradio as gr
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
load_8bit: bool = True
base_model: str = "beomi/KoAlpaca-Polyglot-5.8B"
lora_weights = "./models/korani_LORA_001"
# lora_weights: str = "./models/korani_LORA_pretrained" ### pretrained
server_name: str = "0.0.0.0"
share_gradio: bool = True
base_model = base_model or os.environ.get("BASE_MODEL", "")
assert (
    base_model
), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
#%%
data_dir = "./assets/data/pretrained"
with open(f"{data_dir}/default_work.json", "r", encoding="utf-8") as f:
    default_work = json.load(f)
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
    score=0,
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
            
    """Manual Saving"""
    try:
        df = pd.read_csv('./flagged/logged.csv', encoding='utf-8', index_col=0)
        new = pd.DataFrame(
            [[instruction, result, float(uos), score]],
            columns=df.columns
        )
        pd.concat([df, new], axis=0).to_csv('./flagged/logged.csv', encoding='utf-8')
    except:
        pd.DataFrame(
            [[instruction, result, float(uos), score]],
            columns=['instruction', 'output', 'uos', 'score']
        ).to_csv('./flagged/logged.csv', encoding='utf-8')
    
    yield result
    # yield prompter.get_response(output)
#%%
gr.close_all()

gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2,
            label="Instruction",
            placeholder="(질문 예시) 중앙구매 담당자는 누구인가요?", 
        ),
        gr.components.Checkbox(
            value=True,
            label="서울시립대학교의 업무분장표에 관한 질문을 하기 위해서는 check 표시를 해주세요!"),
        gr.components.Slider(
            minimum=-1, maximum=1, step=1, value=0, label="대답에 따라서 점수를 주세요! (-1점, 0점, 1점)"
        ),
        # gr.components.Slider(
        #     minimum=1, maximum=3, step=1, value=1, label="여러개의 답변을 듣고 싶으신 경우 값을 선택해주세요! (1개 ~ 3개)"
        # ),
        # gr.components.Textbox(
        #     lines=2, label="Input", placeholder="none"),
        # gr.components.Slider(
        #     minimum=0, maximum=1, value=0.1, label="Temperature"
        # ),
        # gr.components.Slider(
        #     minimum=0, maximum=1, value=0.9, label="Top p"
        # ),
        # gr.components.Slider(
        #     minimum=0, maximum=100, step=1, value=5, label="Top k"
        # ),
        # gr.components.Slider(
        #     minimum=1, maximum=4, step=1, value=2, label="Beams"
        # ),
        # gr.components.Slider(
        #     minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        # ),
        # gr.components.Checkbox(label="Stream output"),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="[서울시립대학교 업무분장표 ChatBot] v.0",
    description="""
    Instruction-tuning을 활용한 Large Language Model 기반의 특정 도메인 맞춤형 챗봇 개발
    """,
    # allow_flagging="auto",
).queue().launch(server_name=server_name, share=share_gradio, server_port=7860)
# ).queue().launch(server_name="localhost", share=share_gradio)
#%%
# if __name__ == "__main__":
#     fire.Fire(main)
#%%
# pd.read_csv("./flagged/logged.csv", encoding='utf-8')
#%%