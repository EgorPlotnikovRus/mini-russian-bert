import random
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader


def get_masked_dataset(dataset, tokenizer, mlm_probability):
    vocab = tokenizer.get_vocab()

    def mask_tokens(example, mlm_probability=mlm_probability):
        tokens = example["tokens"]

        if len(tokens) != 512:
            return None

        labels = tokens.copy()

        for i in range(len(tokens)):
            if random.random() < mlm_probability:
                tokens[i] = vocab['[MASK]']

        return {
            "input_ids": tokens,
            "labels": labels
        }

    dataset = dataset.filter(lambda x: len(x["tokens"]) == 512)

    return dataset.map(lambda x: mask_tokens(x))

def split_dataset(dataset, train_size=0.1, val_size=0.05, seed=42):
    small_dataset = dataset.train_test_split(train_size=train_size + val_size, seed=seed)

    train_val_split = small_dataset["train"].train_test_split(train_size=train_size / (train_size + val_size), seed=seed)

    return DatasetDict({
        "train": train_val_split["train"],
        "val": train_val_split["test"]
    })


def get_split_dataset(dataset, tokenizer, train_size, val_size, mlm_probability):

    masked_dataset = get_masked_dataset(dataset, tokenizer, mlm_probability)
    dataset = split_dataset(masked_dataset, train_size=train_size, val_size=val_size)

    return dataset

def collate_fn(batch):

    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    return {"input_ids": input_ids,
            "labels": labels}

def get_dataloaders(dataset, tokenizer, train_size, val_size, train_batch_size, val_batch_size, mlm_probability):

    splited_dataset = get_split_dataset(dataset, tokenizer, train_size, val_size, mlm_probability)
    train_dataloader = DataLoader(splited_dataset['train'], batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(splited_dataset['val'], batch_size=val_batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader