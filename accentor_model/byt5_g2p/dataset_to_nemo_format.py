import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

UKRAINIAN_LETTERS = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
VOCAB = UKRAINIAN_LETTERS + UKRAINIAN_LETTERS.upper() + " "


def clean_text(text: str) -> str:
    """
    Converts text to lowercase and removes all characters
    not in the Ukrainian alphabet or space.
    """
    text = text.lower()
    return ''.join(char for char in text if char in VOCAB)


def clean_labels(text: str) -> str:
    """
    Converts text to lowercase and removes all characters
    not in the Ukrainian alphabet or space.
    """
    text = text.lower()
    return ''.join(char for char in text if char in VOCAB + "+")


def prepare_dataset(df, eval_size=0.01):
    # Clean and rename columns for both dataframes
    df["text"] = df["text"].str.strip()
    df["labels"] = df["labels"].str.strip()

    df["text"] = df["text"].apply(clean_text)
    df["labels"] = df["labels"].apply(clean_labels)

    df.rename(columns={"text": "text_graphemes", "labels": "text"}, inplace=True)

    df_train, df_eval = train_test_split(df, test_size=eval_size, random_state=42)

    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
    eval_dataset = Dataset.from_pandas(df_eval).remove_columns(["__index_level_0__"])

    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


# Load data
df = pd.read_csv("../data/voa_stressed_cleaned_data.csv")

# Prepare dataset
dataset = prepare_dataset(df)

# Save to JSON
dataset["train"].to_json("train.json", lines=True, force_ascii=False)
dataset["eval"].to_json("eval.json", lines=True, force_ascii=False)
