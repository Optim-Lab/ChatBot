import os
import json
import argparse
import numpy as np
from typing import List
from datetime import datetime


def build_cfgs(root="."):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-name-or-path", dest="base_model_name_or_path", type=str)
    parser.add_argument("--bias", dest="bias")
    parser.add_argument("--enable-lora", dest="enable_lora")
    parser.add_argument("--fan-in-fan-out", dest="fan_in_fan_out")
    parser.add_argument("--inference-mode", dest="inference_mode")
    parser.add_argument("--init-lora-weights", dest="init_lora_weights")
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float)
    parser.add_argument("--merge-weights", dest="merge_weights")
    parser.add_argument("--modules-to-save", dest="modules_to_save")
    parser.add_argument("--peft-type", dest="peft_type")
    parser.add_argument("--r", dest="r")
    parser.add_argument("--target-modules", dest="target_modules")
    parser.add_argument("--task-type", dest="task_type")
    parser.add_argument("--data-path", dest="data_path", type=str)
    parser.add_argument("--output-dir", dest="output_dir", type=str)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--micro-batch-size", dest="micro_batch_size", type=int)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--cutoff-len", dest="cutoff_len", type=int)
    parser.add_argument("--val-set-size", dest="val_set_size", type=int)
    parser.add_argument("--train-on-inputs", dest="train_on_inputs", type=bool)
    parser.add_argument("--add-eos-token", dest="add_eos_token", type=bool)
    parser.add_argument("--group-by-length", dest="group_by_length", type=bool)
    parser.add_argument("--wandb-project", dest="wandb_project", type=str)
    parser.add_argument("--wandb-run-name", dest="wandb_run_name", type=str)
    parser.add_argument("--wandb-watch", dest="wandb_watch", type=str)
    parser.add_argument("--wandb-log-model", dest="wandb_log_model", type=str)
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint", type=str)
    parser.add_argument("--prompt-template-name", dest="prompt_template_name", type=str)
    parser.add_argument("--seed", dest="seed", type=int)
    parser.add_argument("--base-config", dest="base_config", type=str, default=f"{root}/assets/config")

    try:
        args = vars(parser.parse_args())
    except:
        args = vars(parser.parse_args([]))

    with open(f"{args['base_config']}/adapter_config.json", "r") as f:
        adapter_cfg = json.load(f)

    with open(f"{args['base_config']}/train_config.json", "r") as f:
        train_cfg = json.load(f)

    adapter_cfg.update({k: v for k, v in args.items() if v is not None})
    train_cfg.update({k: v for k, v in args.items() if v is not None})

    return adapter_cfg, train_cfg


def check_latest_version(data_dir: str) -> List:
    version_list = [ver for ver in os.listdir(data_dir) if ver.__contains__("v.")]
    version_checker = [0,0,0]
    for ver in version_list:
        _, first, second, third = ver.split(".")
        if int(first) >= version_checker[0] & int(second) >= version_checker[1] & int(third) >= version_checker[2]:
            version_checker = [first, second, third]

    return version_checker


def make_new_dir(data_root: str=".") -> str:
    data_dir = f"{data_root}/assets/data"
    version_checker = check_latest_version(data_dir)
    latest_version = f"v.{'.'.join(version_checker)}"

    if "released" in os.listdir(f"{data_dir}/{latest_version}"):
        version_checker[2] = str(int(version_checker[2]) + 1)
        new_version = f"v.{'.'.join(version_checker)}"
        version_dir = f"{new_version}"
    else: 
        version_dir = f"{latest_version}"

    np.random.seed(datetime.now().microsecond)
    exp_name = f"expt_{datetime.now().strftime('%y%m%d%H%M%S')}_{np.random.randint(10000, 99999)}"
    sub_dir = f"{version_dir}/{exp_name}"

    os.makedirs(f"{data_dir}/{sub_dir}", exist_ok=True)
    print(f"Save directory generated: {sub_dir}")

    return data_dir, sub_dir
