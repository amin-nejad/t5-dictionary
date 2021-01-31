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
import sys
sys.path.insert(0, 'src')
from src.model import T5Finetuner
from time import gmtime, strftime
from pytorch_lightning.metrics.functional.nlp import bleu_score
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import yaml
```

```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="t5-dictionary")
```

```python
wandb_logger.experiment.name
```

```python
b = [1,2,3]
a = ['a', 'b', 'c']
c = [4,5,6]
```

```python
[[i,j] for i,j in zip(b,c)]
```

```python
d = dict(zip(a, [[i,j] for i,j in zip(b,c)]))
```

```python
df = pd.DataFrame.from_dict(d, orient='index', columns=['pred', 'targ'])
```

```python
df.index = df.index.set_names(['foo'])
```

```python
df.reset_index()
```

```python
wandb.Table(dataframe=df)
```

```python
data = pd.read_csv("data/dictionary.csv")
```

```python
counts = data.target_text.apply(lambda x: len(x)).value_counts()
```

```python
counts[counts.index > 256].sum()
```

```python
counts[counts.index <= 256].sum()
```

```python
len(data)
```

```python
with open("config.yaml", "r") as stream:
    cfg = yaml.safe_load(stream)

hparams=cfg["hyperparameters"]
```

```python
model = T5Finetuner.load_from_checkpoint(
#     "t5-dictionary/1crc67p7/checkpoints/epoch=1.ckpt",
    "t5-dictionary/1shmk59z/checkpoints/epoch=0.ckpt",
    path_to_dataset="data/dictionary.csv",
    hparams=hparams,
    model_name="t5-small"
)

```

```python
text = "define: twonk"
target = data.iloc[data.index[data.input_text == text][0]][1]
target
```

```python
tokens = model.tokenizer.encode("define: blah", return_tensors="pt")
tokens2 = model.tokenizer.encode("define: blaf", return_tensors="pt")
```

```python
import torch
```

```python
a=torch.stack([tokens, tokens2])
```

```python
a.squeeze()
```

```python
tokens = model.tokenizer.encode(text, return_tensors="pt")
output = model(a.squeeze())
output
```

```python
[output[0].split()]
```

```python
bleu_score([output[0].split()], [target.split()])
```

```python
bleu_score([['cat', 'on', 'the', 'mat']], [[['testing', 'on', 'where'], ['testing', 'here']]])
```

```python
translate_corpus = ['a cat is on blah b'.split()]
reference_corpus = [['there is a cat is on the mat'.split()]]
bleu_score(translate_corpus, reference_corpus).item()
```

```python
torch.mean(torch.stack([torch.tensor(1.), torch.tensor(2.)]))
```

```python
reference_corpus
```

```python
[output[0].split()]
```

```python
[[['testing', 'on', 'where'], ['testing', 'here']]]
```

```python
reference_corpus
```

```python

```
