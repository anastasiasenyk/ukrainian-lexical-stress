# %%
import os
import json
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

# %%
import datasets
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets

from sklearn.model_selection import train_test_split

import numpy as np

UKRAINIAN_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINAIN_VOWELS = "аеєиіїоуюя"
ENGLISH_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# %%
allowed_punctuation = """ .,!?;:'"«»()+-—–"""
other_punctuation = """ $%&<>{}[]*"""

voa_df = pd.read_csv(
    "../accentor_model/byt5/data/voa_stressed_cleaned_data.csv"
)  # './voa_stressed_cleaned_data.csv')
unique_letters = set("".join(voa_df["text"].to_list()))

unique_letters = (
    unique_letters
    - set(UKRAINIAN_LETTERS)
    - set(UKRAINIAN_LETTERS.upper())
    - set(allowed_punctuation)
    - set(other_punctuation)
)

df = voa_df[~voa_df["text"].apply(lambda x: any(c in unique_letters for c in x))]
df = df[["text", "labels"]]
df["text"] = df["text"].str.strip()
df["labels"] = df["labels"].str.strip()
df = df.rename(columns={"text": "text_graphemes", "labels": "text"})
df.shape


# %%
def get_data(df):
    train_df, eval_df = train_test_split(df, test_size=0.01, random_state=42)

    train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
    eval_dataset = Dataset.from_pandas(eval_df).remove_columns(["__index_level_0__"])
    dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    return dataset


dataset = get_data(df)
# %%
dataset
# %%
dataset["train"][20]

# %%
dataset["train"].to_json("train.json", lines=True, force_ascii=False)
dataset["eval"].to_json("eval.json", lines=True, force_ascii=False)
