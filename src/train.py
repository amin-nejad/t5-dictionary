"""Training script."""
import fire
import pytorch_lightning as pl
from model import T5Finetuner


def main(path_to_dataset: str = "data/dictionary.csv", model_name: str = None):
    """Training script."""

    model = T5Finetuner(
        path_to_dataset=path_to_dataset,
        model_name=model_name,
    )

    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(main)
