import torch
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def build_model(
    base_model,
    device_map,
    lora_r,
    lora_alpha,
    lora_target_modules,
    lora_dropout,
    bias,
    task_type,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # special_tokens_dict = {
    #     'additional_special_tokens': ["<|업무분장표|>"] + tokenizer.all_special_tokens}
    # tokenizer.add_special_tokens(special_tokens_dict)

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    model = get_peft_model(model, config)

    return model, tokenizer
