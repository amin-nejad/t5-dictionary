"""Training script."""
from time import gmtime, strftime

import fire
import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger

from model import T5Finetuner


def main(path_to_dataset: str = "data/dictionary.csv", model_name: str = "t5-small"):
    """Training script."""

    with open("config.yaml", "r") as stream:
        cfg = yaml.load(stream)

    hparams = cfg["hyperparameters"]
    num_gpus = cfg["NUM_GPUS"]

    wandb_logger = WandbLogger(
        project="t5-dictionary",
        name=f"t5-dictionary-{strftime('%Y%m%d-%H%M%S', gmtime())}",
    )
    wandb_logger.log_hyperparams(hparams)

    pl.seed_everything(hparams["seed"])

    model = T5Finetuner(
        path_to_dataset=path_to_dataset,
        hparams=hparams,
        model_name=model_name,
    )

    trainer = pl.Trainer(
        gpus=num_gpus,
        accelerator="ddp",
        max_epochs=hparams.MAX_EPOCHS,
        logger=wandb_logger,
        precision=16,
    )
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(main)
