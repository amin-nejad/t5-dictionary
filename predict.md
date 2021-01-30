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
import yaml
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
    "t5-dictionary/1crc67p7/checkpoints/epoch=1.ckpt",
#     "t5-dictionary/3g4wegxm/checkpoints/epoch=0.ckpt",
    path_to_dataset="data/dictionary.csv",
    hparams=hparams,
    model_name="t5-large"
)

```

```python
text = "define: twat"
target = data.iloc[data.index[data.input_text == text][0]][1]
target
```

```python
tokens = model.tokenizer.encode(text, return_tensors="pt")
output = model(tokens)
output
```

```python
tokens
```

```python
model.tokenizer.encode(target, return_tensors="pt")
```

```python
target.split()
```

```python
output[0].split()
```

```python
bleu_score([output[0].split()], [target.split()])
```

```python
bleu_score([['cat', 'on', 'the', 'mat']], [[['testing', 'on', 'where'], ['testing', 'here']]])
```

```python
translate_corpus = ['cat is on mat'.split()]
reference_corpus = [['a cat is on the mat'.split(), 'a cat is on the mat'.split()]]
bleu_score(translate_corpus, reference_corpus)
```

```python
[['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
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
