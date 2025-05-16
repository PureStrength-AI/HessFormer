import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Sequence, Union
from transformers import PreTrainedTokenizerBase

def _tokenise(texts, tokenizer, max_length):
    return tokenizer(
        texts, truncation=True, padding="max_length", max_length=max_length)

def build_dataloader(
    dataset: Union[str, Dataset, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 2,
    max_length: int = 64,
):
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, split="train[:1%]")
    elif isinstance(dataset, (list, tuple)):
        dataset = {"text": list(dataset)}
    # map
    def tok(batch):
        out = _tokenise(batch["text"], tokenizer, max_length)
        return out
    tokenised = dataset.map(tok, batched=True)
    tokenised.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # collate
    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        labels = [x.clone() for x in ids]
        pad_id = tokenizer.pad_token_id
        padded_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = torch.stack([b["attention_mask"] for b in batch])
        return dict(input_ids=padded_ids, labels=padded_labels, attention_mask=attn)
    return DataLoader(tokenised, batch_size=batch_size, collate_fn=collate)
