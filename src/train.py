"""Training script."""
from time import gmtime, strftime

import fire
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

import wandb

from .model import T5Finetuner


def main(path_to_dataset: str = "data/dictionary.csv", model_name: str = None):
    """Training script."""

    with open("config.yaml", "r") as stream:
        cfg = yaml.load(stream)
    wandb.init(
        project=f"t5-dictionary-{strftime('%Y%m%d-%H%M%S', gmtime())}",
        config=cfg["hyperparameters"],
    )

    num_gpus = cfg["NUM_GPUS"]

    torch.manual_seed(wandb.config.SEED)
    np.random.seed(wandb.config.SEED)
    torch.backends.cudnn.deterministic = True

    model = T5Finetuner(
        path_to_dataset=path_to_dataset,
        config=wandb.config,
        model_name=model_name,
    )

    trainer = pl.Trainer(
        gpus=num_gpus, accelerator="ddp", max_epochs=wandb.config.MAX_EPOCHS
    )
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(main)
