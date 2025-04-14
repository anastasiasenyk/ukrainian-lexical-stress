import pandas as pd

df = pd.read_csv('./voice_of_america/voa_stressed_cleaned_data.csv')

UKRAINIAN_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINIAN_VOWELS = "аеєиіїоуюя"
ENGLISH_LETTERS = "abcdefghijklmnopqrstuvwxyz"
allowed_punctuation = """ .,!?;:'’"«»()+-—–0123456789""" + """ $%&<>{}[]*…"""

banned_symbols = set("ыэё")

unique_letters = set("".join(df['text'].tolist()))

unique_letters = (
    unique_letters
    - set(UKRAINIAN_LETTERS)
    - set(UKRAINIAN_LETTERS.upper())
    - set(ENGLISH_LETTERS)
    - set(ENGLISH_LETTERS.upper())
    - set(allowed_punctuation)
)

df = df[~df["text"].apply(lambda x: any(c in banned_symbols for c in x))]

def clean_text(text):
    for symbol in unique_letters:
        text = text.replace(symbol, '')
    return text

df["text"] = df["text"].apply(lambda x: clean_text(x))

df['text'] = df['text'].str.strip()
df['labels'] = df['labels'].str.strip()

df.to_csv('./voa_stressed_cleaned_data.csv', index=False)