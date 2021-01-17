# T5-Dictionary

Repo for experimenting with building a T5 model of the English Dictionary.

## Requirements

Create a virtual environment with python 3.8. To install the requirements, run `pip install -r requirements.txt`.

If modifying the `requirements.in` file, you can generate a new `requirements.txt` by running `pip install pip-tools && pip-compile requirements.in`

## Wandb

The training script utilises [wandb](https://www.wandb.com) for tracking experiments. Create a free account and then use the command below to login from the command line. Once you've done this, experiments will be accessible and trackable in your online portal.

```bash
wandb login
```

## Data

Data is a direct copy of the `data` directory from the fantastic [wordset dictionary](https://github.com/wordset/wordset-dictionary).

## Preprocessing data

Format and preprocess the data as a csv file:

```bash
python src/preprocess.py
```

## Hyperparameters

These are defined in the `.env` file.

## Training

```bash
python src/train.py
```

Arguments can optionally be provided e.g. `--model_name=t5-large`

## Deployment

This is done using `cortex`.

First follow the instructions [here](https://docs.cortex.dev/clusters/gcp/credentials) to create new service, download the key and set it as an environment variable.

Then, spin up the cluster:

```bash
cd cortex && cortex cluster-gcp up --config=cluster.yaml
```

This will spin up a cluster. To create the API and run an example, run the `consume.py` file.

To spin down the cluster:

```bash
 cortex cluster-gcp down
```

### Debugging

`cortex get <api-name>`

or

`cortex logs <api-name>`
