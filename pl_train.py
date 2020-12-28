import pytorch_lightning as pl
from pl import T5Finetuner

model = T5Finetuner(
    path_to_dataset="dictionary.csv",
    model_name="t5-small",
)

trainer = pl.Trainer()
trainer.fit(model)
