import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb
from torch import cuda
from dataset import CustomDataset

device = "cuda" if cuda.is_available() else "cpu"


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]
            if _ % 100 == 0:
                print(f"Completed {_}")

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def main():

    wandb.init(project="t5-dictionary")
    config = wandb.config
    config.TRAIN_BATCH_SIZE = 2
    config.VALID_BATCH_SIZE = 2
    config.TRAIN_EPOCHS = 2
    config.LEARNING_RATE = 1e-4
    config.SEED = 42
    config.MAX_LEN = 512
    config.SUMMARY_LEN = 512

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained("t5-base", truncation=True)

    df = pd.read_csv("dictionary.csv")
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=config.SEED)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(
        train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN
    )
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    training_loader = DataLoader(training_set, batch_size=config.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=config.VALID_BATCH_SIZE)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    wandb.watch(model, log="all")
    print("Initiating Fine-Tuning for the model on our dataset")

    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv("./models/predictions.csv")


if __name__ == "__main__":
    main()