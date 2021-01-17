---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.2"
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from src.model import T5Finetuner
from time import gmtime, strftime

import wandb
import yaml
```

```python
with open("config.yaml", "r") as stream:
    cfg = yaml.load(stream)
wandb.init(project=f"t5-dictionary-{strftime('%Y%m%d-%H%M%S', gmtime())}", config=cfg["hyperparameters"])

```

```python
model = T5Finetuner()
```

```python
model = T5Finetuner.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=1.ckpt", path_to_dataset="data/dictionary.csv", config=wandb.config)

model.eval()

```

```python
payload = {
    "text": "define: retard"
}
```

```python
model.tokenizer.encode(payload["text"])
```

```python
model.tokenizer.convert_ids_to_tokens(model.tokenizer.encode(payload["text"]))
```

```python
tokens = model.tokenizer.encode(payload["text"], return_tensors="pt")
```

```python
tokens
```

```python
output = model.model.generate(
    input_ids=tokens,
    max_length=512,
    num_beams=2,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True,
)
```

```python
model.tokenizer.decode(output[0], skip_special_tokens=True)
```

```python

```
