#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import re
import json
from tqdm import tqdm
import random
#%%
"""
data load
"""
data_dir = "./assets/data"
filename = "inst_synonym"
with open(f"{data_dir}/{filename}.json", "r", encoding="utf-8") as f:
    inst_synonym = json.load(f)

filename = "default_output"
with open(f"{data_dir}/{filename}.json", "r", encoding="utf-8") as f:
    default_output = json.load(f)
#%%
def mask_per_word(work: str) -> str:
    if len(work.split()) <= 2:
        return work
    if len(work.split()) > 2:
        r = 1
    if len(work.split()) > 4:
        r = random.randint(1, 2)
    if len(work.split()) > 8:
        r = random.randint(2, 3)
    sorted_sample = [
        work.split()[i] 
        for i in sorted(random.sample(range(len(work.split())), len(work.split()) - r))
    ]
    return ' '.join(sorted_sample)

def mask_per_space(work: str) -> str:
    work_ = ""
    for x in work.split():
        work_ += x
        if random.uniform(0, 1) < 0.9:
            work_ += " "
    return work_.strip()
#%%
"""
Feature Engineering - augmentation
"""
random.seed(42)
augmented = {}
for i, (name, inst) in tqdm(enumerate(inst_synonym.items()), desc="Masking..."):
    all_inst = []
    for j in range(len(inst)):
        all_inst += [inst[j]]
        
        if len(inst[j].split()) > 1:
            r = len(inst[j].split())
            for _ in range(int(r * 2)): # random repeat number (hyperparameter)
                # drop word
                work = mask_per_word(inst[j])
                # drop space
                work = mask_per_space(work)
                
                work = re.sub(' +', ' ', work)
                if len(work) > 1:
                    all_inst.append(work)
    augmented[name] = all_inst

with open(f"{data_dir}/augmented_work.json", "w", encoding="utf-8") as f:
    json.dump(augmented, f, ensure_ascii=False, indent=4)

print(f"Augmented samples number: {len(augmented)}")
#%%
augmented
#%%
def generate_templates(inst): # random sampling
    return random.sample([
        f"{inst}을 담당하는 분은 누구입니까?",
        f"{inst}를 담당하는 사람은 누구인가요?",
        f"{inst}의 담당자는 누구신가요?",
        f"{inst}을 맡고 계신 분은 누구세요?",
        f"누가 {inst}를 담당하고 있나요?",
        f"{inst}을 책임지고 있는 분은 누구인가요?",
        f"{inst}를 관리하고 계신 분은 누구입니까?",
        f"{inst}을 전담하고 있는 사람은 누구세요?",
        f"누가 {inst}을 책임지고 있나요?",
        f"{inst}에 대한 책임자는 누구로 되어 있나요?",
        f"누가 {inst}을 관리하고 있나요?",
        f"{inst}을 책임지고 계신 사람은 누구신지요?"
    ], 1)
#%%
"""
Feature Engineering - build instruction dictionary using 9 basic templates
"""
with open(f"{data_dir}/augmented_work.json", "r", encoding="utf-8") as f:
    augmented = json.load(f)

training_dict = {}
for i, (name, inst) in tqdm(enumerate(augmented.items())):
    all_inst = []
    for j in range(len(inst)):
        all_inst.append(generate_templates(inst[j])[0])
    training_dict[name] = all_inst

with open(f"{data_dir}/training_dict.json", "w", encoding="utf-8") as f:
    json.dump(training_dict, f, ensure_ascii=False, indent=4)
#%%
training_dict
#%%
"""
Feature Engineering - build instruction-tuning dataset using 9 basic templates
"""
training_data = []
for i, (name, inst) in tqdm(enumerate(augmented.items())):
    all_inst = []
    for j in range(len(inst)):
        all_inst.append(generate_templates(inst[j])[0])
    
    for inst in all_inst:
        synth_sample = {}
        synth_sample["instruction"] = inst.strip()
        synth_sample["input"] = ""
        synth_sample["output"] = default_output.get(name).strip()
        training_data.append(synth_sample)

with open(f"{data_dir}/training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=4)
    
print(f"Initial samples number: {len(training_data)}")
#%%
training_data
#%%