{
  "base_model_name_or_path": "beomi/KoAlpaca-Polyglot-5.8B",
  "bias": "none",
  "enable_lora": null,
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "merge_weights": false,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,
  "target_modules": [
    "query_key_value",
    "xxx"
  ],
  "task_type": "CAUSAL_LM"
}
{
  "data_path": "./assets/data/정책기획팀_신유정_00.json",
  "output_dir": "./models",
  "batch_size": 128,
  "micro_batch_size": 4,
  "num_epochs": 20,
  "learning_rate": 3e-3,
  "cutoff_len": 256,
  "val_set_size": 0,
  "train_on_inputs": false,
  "add_eos_token": false,
  "group_by_length": false,
  "wandb_project": "KorANI",
  "wandb_run_name": "",
  "wandb_watch": "",
  "wandb_log_model": "false",
  "resume_from_checkpoint": null,
  "prompt_template_name": "korani",
  "seed": 42
}
