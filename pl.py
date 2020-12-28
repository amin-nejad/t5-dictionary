import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb
from torch import cuda
from dataset import CustomDataset
import time


class T5Finetuner(pl.LightningModule):
    def __init__(self, path_to_dataset: str, model_name="t5-small"):

        super().__init__()

        wandb.init(project=model_name + f"-{time.time()}")
        config = wandb.config
        config.TRAIN_BATCH_SIZE = 2
        config.VALID_BATCH_SIZE = 2
        config.TRAIN_EPOCHS = 2
        config.LEARNING_RATE = 1e-4
        config.SEED = 42
        config.MAX_LEN = 512
        config.OUTPUT_LEN = 512

        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        torch.backends.cudnn.deterministic = True

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, truncation=True)

        df = pd.read_csv(path_to_dataset)
        train, validate, test = np.split(df, [int(0.8 * len(df)), int(0.9 * len(df))])

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.training_set = CustomDataset(
            train.reset_index(drop=True),
            self.tokenizer,
            config.MAX_LEN,
            config.OUTPUT_LEN,
        )
        self.val_set = CustomDataset(
            validate.reset_index(drop=True),
            self.tokenizer,
            config.MAX_LEN,
            config.OUTPUT_LEN,
        )
        self.test_set = CustomDataset(
            test.reset_index(drop=True),
            self.tokenizer,
            config.MAX_LEN,
            config.OUTPUT_LEN,
        )

        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            lm_labels=lm_labels,
        )

    def training_step(self, batch, batch_idx):

        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = (
            batch["source_ids"],
            batch["source_mask"],
            batch["target_ids"],
        )
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=y_ids,
            lm_labels=lm_labels,
        )
        loss = outputs[0]

        return loss

    def validation_step(self, batch, batch_idx):

        source_ids, source_mask, y = (
            batch["source_ids"],
            batch["source_mask"],
            batch["target_ids"],
        )

        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in y
        ]

        return preds, target

    def test_step(self, batch, batch_idx):

        source_ids, source_mask, y = (
            batch["source_ids"],
            batch["source_mask"],
            batch["target_ids"],
        )

        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in y
        ]

        return preds, target

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.model.parameters(), lr=self.config.LEARNING_RATE
        )

    def train_dataloader(self):

        return DataLoader(self.training_set, batch_size=self.config.TRAIN_BATCH_SIZE)

    def val_dataloader(self):

        return DataLoader(self.val_set, batch_size=self.config.VALID_BATCH_SIZE)

    def test_dataloader(self):

        return DataLoader(self.test_set, batch_size=self.config.VALID_BATCH_SIZE)