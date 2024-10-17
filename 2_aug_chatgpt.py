#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
from tqdm import tqdm
import random
from pprint import pprint
from joblib import Parallel, delayed

from module.synonym import generate_synonym
#%%
"""
Data Load
"""
data_dir = "./assets/data"
filename = "default_work"
with open(f"{data_dir}/{filename}.json", "r", encoding="utf-8") as f:
    default_work = json.load(f)
#%%
name, inst = list(default_work.items())[0]
inst_split = [x[:-1] for x in inst.split('- ') if len(x)] 
syn_list, response = generate_synonym(inst_split[0])
response
#%%
"""
Feature Engineering - build synonym
"""
num_process = 16 ### number of process for parallel version
inst_synonym = {}
for i, (name, inst) in tqdm(
    enumerate(list(default_work.items())[:5]), # 빠른 확인을 위한 변경 (추후 수정)
    desc="Build instructions with synonyms..."):
    
    inst_split = [x[:-1] for x in inst.split('- ') if len(x)] # remove "\n"
    
    ### For loop version
    results = []
    for x in inst_split:
        results.append(generate_synonym(x))
    
    ### Parallel version
    # results = Parallel(n_jobs=num_process)(
    #     delayed(generate_synonym)(x) for x in inst_split)
    
    inst_synonym[name] = sum([[x] + y for x, y in zip(inst_split, results)], [])

    # 중간 결과 확인
    if i % 5 == 0 and i != 0:
        pprint(random.sample(list(inst_synonym.items()), 5))

filename = f'inst_synonym'
with open(f"{data_dir}/{filename}.json", "w", encoding="utf-8") as f:
    json.dump(inst_synonym, f, ensure_ascii=False, indent=4)
#%%
inst_synonym
#%%