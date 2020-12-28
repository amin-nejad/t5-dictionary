"""Preprocessing script.

Uses vaex to process data and output a 'dictionary.csv' file in `data`.
The file contains two columns: 'input_text' and 'output_text'.
"""
# %%
import string

import matplotlib.pyplot as plt
import pandas as pd
import vaex
from tqdm import tqdm


# %%
def preprocess_meanings(meanings: str) -> str:
    """Preprocess meanings.

    Outputs a string of the form:

        <NUMBER> <PART_OF_SPEECH> <DEFINTION> ...

        e.g.

        1 <speech_part> noun <def> this is a definition \
            2 <speech_part> noun <def> another definition etc.

    Args:
        definitions (str): JSON of word definitions as a string.

    Returns:
        str: preprocessed string of definitions
    """
    concatenated_meanings = []
    for i, meaning in enumerate(meanings):
        concatenated_meanings.append(str(i + 1))

        # PART OF SPEECH
        speech_part = meaning.get("speech_part")
        if speech_part:
            concatenated_meanings.append("<speech_part>")
            concatenated_meanings.append(speech_part)

        # DEFINITION
        definition = meaning.get("def")
        if definition:
            concatenated_meanings.append("<def>")
            concatenated_meanings.append(definition)

        # EXAMPLE
        example = meaning.get("example")
        if example:
            concatenated_meanings.append("<example>")
            concatenated_meanings.append(example)

    return " ".join(concatenated_meanings)


# %%
# Get list of input files
list_of_df_names = list(string.ascii_lowercase)
list_of_df_names.append("misc")


# %%
# Loop through input files and preprocess the meanings


full_dictionary = pd.DataFrame()
print("Preprocessing definitions")

for df_name in tqdm(list_of_df_names):
    df = vaex.from_json("data/" + df_name + ".json", orient="index")
    df = df.dropna(["meanings"])
    df["target_text"] = df.meanings.apply(preprocess_meanings)
    df["input_text"] = "define: " + df.word
    df = df[["input_text", "target_text"]]
    full_dictionary = pd.concat([full_dictionary, df.to_pandas_df()], ignore_index=True)


# %%
# Shuffle dataset and save
print(f"Dataset length: {len(full_dictionary)}")
full_dictionary = full_dictionary.sample(frac=1).reset_index(drop=True)
full_dictionary.to_csv("data/dictionary.csv", index=False)
print("Pre-processed data saved to 'data/dictionary.csv'")

# %% [markdown]
# Some insights into our data:

# %%
df["target_len"] = df.target_text.apply(lambda x: len(x))
df["target_num_defs"] = df.target_text.apply(lambda x: x.count("<def>"))

# %%
plt.hist(df["target_len"].tolist())
# %%
plt.hist(df["target_num_defs"].tolist())
