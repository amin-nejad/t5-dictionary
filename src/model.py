"""Model defined here."""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional.nlp import bleu_score
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dataset import CustomDataset


class T5Finetuner(pl.LightningModule):
    """Pytorch Lightning module for fine-tuning T5 model.

    Args:
        path_to_dataset (str): string path to dataset
        hparams (Dict): dictionary containing hyperparameters
        model_name (str, optional): t5 model name
    """

    def __init__(
        self,
        path_to_dataset: str,
        hparams: Dict[str, Union[float, int]],
        model_name: str,
    ):

        super().__init__()

        self.hparams = hparams
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, truncation=True)
        self.tokenizer.add_tokens(["<speech_part>", "<def>", "<example>"])

        df = pd.read_csv(path_to_dataset)
        df_train, df_validate = np.split(df, int(0.9 * len(df)))

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.training_set = CustomDataset(
            df_train.reset_index(drop=True),
            self.tokenizer,
            self.hparams["MAX_INPUT_LEN"],
            self.hparams["MAX_OUTPUT_LEN"],
        )
        self.val_set = CustomDataset(
            df_validate.reset_index(drop=True),
            self.tokenizer,
            self.hparams["MAX_INPUT_LEN"],
            self.hparams["MAX_OUTPUT_LEN"],
        )

    def forward(self, source_ids: torch.Tensor, source_mask: torch.Tensor = None):
        """Forward method for performing inference."""

        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=self.hparams["MAX_OUTPUT_LEN"],
            num_beams=self.hparams["NUM_BEAMS"],
            repetition_penalty=self.hparams["REPETITION_PENALTY"],
            length_penalty=self.hparams["LENGTH_PENALTY"],
            early_stopping=True,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]

        return preds

    def _get_loss(self, source_ids, source_mask, y) -> float:
        """Returns loss. Used by both training and validation loops."""

        pad_token_id = self.tokenizer.pad_token_id
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()

        # Replace pad token id with -100
        # See https://github.com/huggingface/transformers/issues/6238
        lm_labels[y[:, 1:] == pad_token_id] = -100

        outputs = self.model(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=y_ids,
            lm_labels=lm_labels,
        )
        loss = outputs[0]
        return loss.item()

    def training_step(self, batch, batch_idx):

        source_ids, source_mask, y = (
            batch["source_ids"],
            batch["source_mask"],
            batch["target_ids"],
        )

        loss = self._get_loss(source_ids, source_mask, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        source_ids, source_mask, y = (
            batch["source_ids"],
            batch["source_mask"],
            batch["target_ids"],
        )

        loss = self._get_loss(source_ids, source_mask, y)

        preds = self(source_ids, source_mask)

        target = [
            self.tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in y
        ]

        bleu_score_val = bleu_score(preds, target)

        return loss, bleu_score_val

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.model.parameters(), lr=self.hparams["LEARNING_RATE"]
        )

    def train_dataloader(self):

        return DataLoader(
            self.training_set,
            batch_size=self.hparams["TRAIN_BATCH_SIZE"],
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_set,
            batch_size=self.hparams["VALID_BATCH_SIZE"],
            num_workers=8,
            pin_memory=True,
        )

    def validation_epoch_end(self, val_step_outputs: List):

        avg_loss = [i[0] for i in val_step_outputs].mean()
        avg_bleu_score = [i[1] for i in val_step_outputs].mean()

        self.log("val_loss", avg_loss)
        self.log("val_bleu", avg_bleu_score)
