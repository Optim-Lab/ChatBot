from datasets import load_dataset
from .preprocessor import generate_and_tokenize_prompt


def split_dataset(
    data_path,
    val_set_size,
    cutoff_len,
    tokenizer,
    prompter,
    train_on_inputs,
    add_eos_token,
    seed=42,
):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(
                    cutoff_len, tokenizer, prompter, train_on_inputs, add_eos_token, x
                )
            )
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(
                    cutoff_len, tokenizer, prompter, train_on_inputs, add_eos_token, x
                )
            )
        )
    else:
        train_data = (
            data["train"]
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(
                    cutoff_len, tokenizer, prompter, train_on_inputs, add_eos_token, x
                )
            )
        )
        val_data = None

    return train_data, val_data
