"""Custom Dataset."""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Dataset):
    """Custom Dataset for our dictionary task.

    Args:
        dataframe (pd.DataFrame): pandas dataframe containing the data
        tokenizer (PreTrainedTokenizer): Tokenizer from Huggingface
        source_len (int): maximum source sequence length
        target_len (int): maximum target sequence length
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_len: int,
        target_len: int,
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.input_text = self.data.input_text
        self.target_text = self.data.target_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        input_text = " ".join(input_text.split())

        target_text = str(self.target_text[index])
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }
