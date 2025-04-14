import os
import re
from collections import defaultdict

import marisa_trie
import pandas as pd


def clean_word(word):
    word = word.lower()
    return re.sub(f"^[^{UKRAINIAN_ALL}+]+|[^{UKRAINIAN_ALL}+]+$", "", word)


UKRAINIAN_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINIAN_ALL = UKRAINIAN_LETTERS + UKRAINIAN_LETTERS.upper()
VOCABULARY = marisa_trie.BytesTrie().load(os.path.join("..", "data", "stress.trie"))

df = pd.read_csv("lexical_stress_dataset.csv")
heteronym_counter = defaultdict(int)

for sentence in df["StressedSentence"]:
    sentence = sentence.replace("+", "").lower()
    words = sentence.split()
    for word in words:
        cleaned = clean_word(word)
        if cleaned in VOCABULARY:
            try:
                if VOCABULARY[cleaned][0].count(b"\xff") > 1:
                    heteronym_counter[cleaned] += 1
            except Exception as e:
                pass

sorted_heteronyms = sorted(heteronym_counter.items())
sorted_heteronyms = pd.DataFrame(sorted_heteronyms, columns=["Heteronym", "Count"])
sorted_heteronyms.to_csv("heteronyms_list.csv")
