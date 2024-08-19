#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import re
import json
import pandas as pd
import numpy as np
#%%
"""
Directory
"""
data_dir = "./assets/data" ### raw data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
#%%
"""
Data Preprocessing
"""
seed_data_dir = "./data/rnr_table.csv"
base_df = pd.read_csv(seed_data_dir, encoding='utf-8')
default_work = {}
default_output = {}
for i in range(len(base_df)):
    title = base_df["부서"].iloc[i] + ', ' + base_df["직위"].iloc[i]
    name = base_df["담당자"].iloc[i]
    phone = base_df["전화번호"].iloc[i]
    jobs = base_df["담당업무"].iloc[i]
    if type(jobs) == float and np.isnan(jobs): continue
    jobs = re.sub("&", "N", jobs) ### 특수문자 제거
    
    output = f"담당자는 {title} '{name}' 입니다. 전화번호는 '{phone}' 입니다. '{name}'의 주요업무는 다음과 같습니다."
    default_output[name] = output
    
    instruction = "".join([x.strip() + "\n" for x in jobs.split("ㆍ") if len(x)]) ### 특수문자 제거
    if instruction == '\n': continue
    default_work[name] = instruction

with open(f"{data_dir}/default_output.json", "w", encoding="utf-8") as f:
    json.dump(default_output, f, ensure_ascii=False, indent=4)
with open(f"{data_dir}/default_work.json", "w", encoding="utf-8") as f:
    json.dump(default_work, f, ensure_ascii=False, indent=4)
#%%
default_output
#%%
default_work
#%%