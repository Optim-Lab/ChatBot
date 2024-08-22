#%%
"""
Reference:
[1] https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import os
import sys
import json
import torch
import transformers

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from module.data_pipeline.prompter import Prompter
from module.data_pipeline.loader import split_dataset
from module.model.lora import build_model
from module.utils import build_cfgs
#%%
def main(
    # model and data
    base_model_name_or_path: str,
    data_path: str,
    output_dir: str,
    # training hyperparams
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    # lora hyperparams
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list,  # workaround to use 8bit training on this model
    # llm hyperparams
    train_on_inputs: bool,  # if False, masks out inputs in loss
    add_eos_token: bool,
    group_by_length: bool,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str, # The wandb project name, if wandb is active
    wandb_run_name: str,
    wandb_watch: str,  # options: false | gradients | all
    wandb_log_model: str,  # options: false | true
    resume_from_checkpoint: str,  # either training checkpoint or final adapter
    prompt_template_name: str,  # The prompt template to use, will default to alpaca.
    bias: str,
    task_type: str,
    seed: int,
    # unknown
    enable_lora,
    fan_in_fan_out,
    inference_mode,
    init_lora_weights,
    merge_weights,
    modules_to_save,
    peft_type,
    base_config,
):
    assert base_model_name_or_path

    folder_idx = 0
    while True:
        save_dir = f"{output_dir}/{prompt_template_name}_{peft_type}_{'{:0>3}'.format(str(folder_idx))}"
        if os.path.exists(save_dir):
            folder_idx += 1
            continue
        break
    os.makedirs(save_dir)

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model, tokenizer = build_model(
        base_model_name_or_path,
        device_map,
        r,
        lora_alpha,
        target_modules,
        lora_dropout,
        bias,
        task_type,
    )

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    train_data, val_data = split_dataset(
        data_path,
        val_set_size,
        cutoff_len,
        tokenizer,
        prompter,
        train_on_inputs,
        add_eos_token,
        seed,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=save_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(save_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")

    return save_dir
#%%
if __name__ == "__main__":
    adapter_cfgs, train_cfgs = build_cfgs()
    configs = {}
    configs.update(adapter_cfgs)
    configs.update(train_cfgs)
    save_dir = main(**configs)

    with open(f"{save_dir}/train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_cfgs, f, ensure_ascii=False, indent=4)
#%%
"""학습 데이터 확인"""
# prompt_template_name = "korani"
# prompter = Prompter(prompt_template_name)

# from transformers import AutoTokenizer
# base_model = "beomi/KoAlpaca-Polyglot-5.8B"
# tokenizer = AutoTokenizer.from_pretrained(base_model)
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# train_data, val_data = split_dataset(
#     data_path="./assets/data/training_data.json",
#     val_set_size=512,
#     cutoff_len=256,
#     tokenizer=tokenizer,
#     prompter=prompter,
#     train_on_inputs=False,
#     add_eos_token=False,
#     seed=42,
# )

# sample = next(iter(train_data)) ### 실제 input

# tokenizer.decode(sample["input_ids"]) ### 실제 input의 decode 결과

# tokenizer.decode([sample["input_ids"][i] for i, x in enumerate(sample["labels"]) if x == -100]) ### LLM에게 주어지는 부분

# tokenizer.decode([x for x in sample["labels"] if x != -100]) ### LLM이 예측해야하는 부분
#%%