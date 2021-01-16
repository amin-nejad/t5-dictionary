"""Training script."""
import time

import fire
import pytorch_lightning as pl
import wandb
import yaml

from src.model import T5Finetuner


def main(path_to_dataset: str = "data/dictionary.csv", model_name: str = None):
    """Training script."""

    with open("config.yaml", "r") as stream:
        cfg = yaml.load(stream)
    wandb.init(project=f"t5-dictionary-{time.time()}", config=cfg["hyperparameters"])

    num_gpus = cfg["NUM_GPUS"]

    model = T5Finetuner(
        path_to_dataset=path_to_dataset,
        config=wandb.config,
        model_name=model_name,
    )

    trainer = pl.Trainer(gpus=list(range(num_gpus)))
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(main)
