#%%
import os
# Set the environment variable to limit visible GPUs
gpu_idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
#%%
import sys
import json

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
# data_dir = "./assets/data"
data_dir = "./assets/data/pretrained" ### pretrained version
with open(f"{data_dir}/default_work.json", "r", encoding="utf-8") as f:
    default_work = json.load(f)
#%%
load_8bit = True
base_model= "beomi/KoAlpaca-Polyglot-5.8B"
lora_weights = "./models/korani_LORA_000"
# lora_weights = "./models/pretrained/korani_LORA_pretrained" ### pretrained

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
"""fine-tuning result"""
instruction = "연구비 관리를 담당하는 분은 누구입니까?"
# instruction = "R&D기반조성사업 관련 문의는 누구에게 해야 하나요?"
batch_size=3 # for multiple answers
input=None

topk=5
topp=0.8
temperature=0.5

max_new_tokens=512
pad_token_id=tokenizer.pad_token_id
eos_token_id=tokenizer.eos_token_id
#%%
"""1. Greedy sampling"""
torch.manual_seed(2) # fixed seed
prompt = prompter_uos.generate_prompt(instruction, input)
inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
sequence = inputs['input_ids'].to(device)

for i in range(max_new_tokens):
    # get the predictions
    with torch.no_grad():
        model.float()
        output = model.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=False
        ) # [batch_size, T, vocab_size] (prediction of next token)
        
    # focus only on the last time step (last generated token)
    logits = output.logits[:, -1, :] # [batch_size, vocab_size]
    # Get the index of the token with the highest probability
    idx_next = torch.argmax(logits, dim=-1).unsqueeze(-1) # [batch_size, 1]
    # append sampled index to the running sequence
    sequence = torch.cat((sequence, idx_next), dim=1) # [batch_size, T+1]
    
    # get updated inputs
    next_inputs = [tokenizer.decode(s) for s in sequence]
    inputs = tokenizer(next_inputs, return_tensors="pt")
    
    if sequence[0, -1] == eos_token_id: ### stopping criterion
        break

### post-processing (extract only response)
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
# outputs = list(set(outputs)) # remove duplicated outputs
    
if len(outputs) > 1:
    result = ""
    for i, output in enumerate(outputs):
        result += f"[답변 {i+1}]\n"
        result += output + "\n\n"
else:
    result = outputs[0]

greedy_result = result
print(greedy_result)
#%%
"""2. top-K sampling"""
torch.manual_seed(2) # fixed seed
prompt = prompter_uos.generate_prompt(instruction, input)
inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
sequence = inputs['input_ids'].to(device)

for i in range(max_new_tokens):
    # get the predictions
    with torch.no_grad():
        model.float()
        output = model.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=False
        ) # [batch_size, T, vocab_size] (prediction of next token)
        
    # focus only on the last time step (last generated token)
    logits = output.logits[:, -1, :] # [batch_size, vocab_size]
    # Only keep top-1 + top-K indices
    topk_logits = torch.cat(
        [
            torch.topk(l, k)[0][[-1]].unsqueeze(0) 
            for l, k in zip(logits, [1] + [topk] * (batch_size-1))
        ], dim=0)
    indices_to_remove = logits < topk_logits
    logits[indices_to_remove] = torch.tensor(float('-inf')).to(device)
    # Convert logits to probabilities
    probabilities = (logits / temperature).softmax(dim=-1).to(device)
    # Sample n=1 tokens from the resulting distribution
    idx_next = torch.multinomial(probabilities, num_samples=1).to(device) # [batch_size, 1]
    # append sampled index to the running sequence
    sequence = torch.cat((sequence, idx_next), dim=1) # [batch_size, T+1]
    
    # get updated inputs
    next_inputs = [tokenizer.decode(s) for s in sequence]
    inputs = tokenizer(next_inputs, return_tensors="pt")
    
    if sequence[0, -1] == eos_token_id: ### stopping criterion
        break

### post-processing (extract only response)
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
# outputs = list(set(outputs)) # remove duplicated outputs
    
if len(outputs) > 1:
    result = ""
    for i, output in enumerate(outputs):
        result += f"[답변 {i+1}]\n"
        result += output + "\n\n"
else:
    result = outputs[0]

top_k_result = result
print(top_k_result)
#%%
"""3. top-p sampling"""
torch.manual_seed(2) # fixed seed
prompt = prompter_uos.generate_prompt(instruction, input)
inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
sequence = inputs['input_ids'].to(device)

for i in range(max_new_tokens):
    # get the predictions
    with torch.no_grad():
        model.float()
        output = model.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=False
        ) # [batch_size, T, vocab_size] (prediction of next token)
        
    # focus only on the last time step (last generated token)
    logits = output.logits[:, -1, :] # [batch_size, vocab_size]
    # Convert logits to probabilities
    """temperature sampling is utilized"""
    probs = (logits / 5).softmax(dim=-1).to(device) # temperature tau = 5
    # probs.max()
    # probs = (logits / 1).softmax(dim=-1).to(device) # temperature tau = 1
    # probs.max()
    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    # get the cumulative sum of probabilities
    cumsum_probs = sorted_probs.cumsum(dim=1).to(device)
    # Mask out tokens that don't belong to the top-p set
    sorted_indices_to_remove = cumsum_probs > topp
    logits[sorted_indices_to_remove] = torch.tensor(float('-inf')).to(device)
    # Convert logits to probabilities
    probabilities = (logits / temperature).softmax(dim=-1).to(device)
    # Sample n=1 tokens from the resulting distribution
    idx_next = torch.multinomial(probabilities, num_samples=1).to(device) # [batch_size, 1]
    # append sampled index to the running sequence
    sequence = torch.cat((sequence, idx_next), dim=1) # [batch_size, T+1]
    
    # get updated inputs
    next_inputs = [tokenizer.decode(s) for s in sequence]
    inputs = tokenizer(next_inputs, return_tensors="pt")
    
    if sequence[0, -1] == eos_token_id: ### stopping criterion
        break

### post-processing (extract only response)
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
# outputs = list(set(outputs)) # remove duplicated outputs
    
if len(outputs) > 1:
    result = ""
    for i, output in enumerate(outputs):
        result += f"[답변 {i+1}]\n"
        result += output + "\n\n"
else:
    result = outputs[0]

top_p_result = result
print(top_p_result)
#%%
"""Check the results"""
print("\n======Greedy sampling:======")
print(greedy_result)

print("\n======top-K sampling:======")
print(top_k_result)

print("\n======top-p sampling:======")
print(top_p_result)
#%%
"""original model result"""
# instruction = "점심 메뉴 추천해줘."
instruction = "딥러닝이 뭐야?"
input=None
max_new_tokens=512
pad_token_id=tokenizer.pad_token_id
eos_token_id=tokenizer.eos_token_id

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

print(result)
#%%