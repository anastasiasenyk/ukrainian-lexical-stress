import re
import difflib
import marisa_trie
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance

tqdm.pandas()

# Constants
UKRAINIAN_LETTERS = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
VOWELS = 'ауеїоєяиюі'
ALL_VOWELS = VOWELS + VOWELS.upper()
SIMILARITY_THRESHOLD = 0.65
TRIE_FILE_PATH = './stress.trie'  # Source: https://github.com/lang-uk/ukrainian-word-stress/blob/main/ukrainian_word_stress/data/stress.trie
VOCABULARY = marisa_trie.BytesTrie().load(TRIE_FILE_PATH)


def preprocess_text(sentence: str) -> str:
    """Cleans a given sentence by removing non-Ukrainian characters and converting to lowercase."""
    words = re.split(r'[ ,.?!;:«»()[\]-]+', sentence.lower())
    pattern = rf'^[^{UKRAINIAN_LETTERS}]+|[^{UKRAINIAN_LETTERS}]+$'
    return " ".join(re.sub(pattern, '', word) for word in words if word)


def compute_sequence_similarity(text1: str, text2: str) -> float:
    """Calculates the SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def compute_levenshtein_similarity(text1: str, text2: str) -> float:
    """Computes Levenshtein similarity between two strings."""
    max_length = max(len(text1), len(text2))
    return 1 - (levenshtein_distance(text1, text2) / max_length) if max_length > 0 else 1


def merge_texts(text1: str, text2: str) -> str:
    """Merges two texts b adding stress marks from second text to first."""
    diff = difflib.ndiff(text1, text2)
    merged_text = [d[2:] for d in diff if not d.startswith('+') or d[2:] == '+']
    return ''.join(merged_text)


def apply_stress_mark(word: str) -> str:
    """Adds a stress mark to a given Ukrainian word based on predefined vocabulary for words with no ambiguity."""
    cleaned_word = word.replace('+', '').lower()
    if cleaned_word not in VOCABULARY:
        cleaned_word = cleaned_word.capitalize()
        if cleaned_word not in VOCABULARY:
            return word  # Return original word if not found

    if VOCABULARY[cleaned_word][0].count(b'\xff') > 1:
        return word  # Return original word if multiple stress marks exist

    if cleaned_word == 'його':
        return 'йог+о'  # Exception handling
    if cleaned_word == 'воду':
        return 'в+оду'  # Exception handling

    word_metadata = VOCABULARY[cleaned_word][0].replace(b'\t', b'')
    if len(word_metadata) != 1:
        return word
    stress_position = ord(word_metadata) - 1
    return cleaned_word[:stress_position] + "+" + cleaned_word[stress_position:]


def correct_stress_marks(sentence: str) -> str:
    """Fixes misplaced or missing stress marks in a sentence."""
    words = re.split(r'(\s+|-)', sentence)  # Split by spaces and hyphens
    corrected_words = []

    for word in words:
        # if several stress marks in sequence - keep only one
        fixed_word = re.sub(r'\++', '+', word)

        # if stress mark after the vowel and before other symbols - shift it left
        fixed_word = re.sub(f'([{ALL_VOWELS}])\+(?![{ALL_VOWELS}])', r'+\1', fixed_word)
        # remove all stress marks where there are no vowels before and after
        fixed_word = re.sub(r'(?<![{}])\+(?![{}])'.format(ALL_VOWELS, ALL_VOWELS), '', fixed_word)
        fixed_word = re.sub(r'\++', '+', fixed_word)

        vowel_count = sum(1 for char in fixed_word if char in ALL_VOWELS)

        # if no stress mark, but one vowels - add stress
        if '+' not in word and vowel_count == 1:
            fixed_word = re.sub(f'([{ALL_VOWELS}])', r'+\1', fixed_word, count=1)

        # if stress mark, but no vowels - remove stress
        if '+' in word and vowel_count == 0:
            fixed_word = fixed_word.replace('+', '')

        if vowel_count > 1 and word.count('+') != 1:
            fixed_word = apply_stress_mark(fixed_word)

        corrected_words.append(fixed_word)

    return ''.join(corrected_words)


def detect_wrong_stresses(sentence: str) -> bool:
    """Detects whether a sentence has incorrect stress marks."""
    words = re.split(r'(\s+|-)', sentence)

    for word in words:
        stresses_in_word = sum(1 for char in word if char == '+')
        if stresses_in_word >= 2:
            return True

        vowels_in_word = sum(1 for char in word if char in ALL_VOWELS)
        if vowels_in_word >= 1 and stresses_in_word == 0:
            return True

    return False


# Load dataset
df = pd.read_csv('./raw/voa_transcribed_236k_rows_stressed.csv')
df = df[~df['wav2vec2_transcription'].isna()]

df.loc[:,'cleaned_transcription'] = df['wav2vec2_transcription'].progress_apply(preprocess_text)
df = df.drop_duplicates(subset=['text'], keep='first')
df = df.drop_duplicates(subset=['cleaned_transcription'], keep='first')

df.loc[:,'sequence_similarity'] = df.progress_apply(
    lambda row: compute_sequence_similarity(row['cleaned_transcription'], row['text']), axis=1)

df.loc[:,'levenshtein_similarity'] = df.progress_apply(
    lambda row: compute_levenshtein_similarity(row['cleaned_transcription'], row['text']), axis=1)

filtered_df = df[(df['sequence_similarity'] >= SIMILARITY_THRESHOLD) |
                 (df['levenshtein_similarity'] >= SIMILARITY_THRESHOLD)]

filtered_df.loc[:,'merged_transcription'] = filtered_df.progress_apply(
    lambda row: merge_texts(row['wav2vec2_transcription'], row['wav2vec_stress_transcription']), axis=1)
filtered_df.loc[:,'merged_transcription_corrected'] = filtered_df['merged_transcription'].progress_apply(correct_stress_marks)
filtered_df.loc[:,'is_incorrect'] = filtered_df['merged_transcription_corrected'].progress_apply(detect_wrong_stresses)


# Do the same process but based on original transcription.
filtered_df.loc[:,'merged_transcription_2'] = filtered_df.progress_apply(
    lambda row: merge_texts(row['text'], row['wav2vec_stress_transcription']), axis=1)
filtered_df.loc[:,'merged_transcription_corrected_2'] = filtered_df['merged_transcription_2'].progress_apply(correct_stress_marks)
filtered_df.loc[:,'is_incorrect_2'] = filtered_df['merged_transcription_corrected_2'].progress_apply(detect_wrong_stresses)
# filtered_df.to_csv('./voa_stressed.csv')

# Apply the condition and update 'merged_transcription_corrected' and 'is_incorrect' where condition is met
filtered_df.loc[
    (filtered_df['is_incorrect'] == True) & (filtered_df['is_incorrect_2'] == False),
    ['merged_transcription_corrected', 'is_incorrect']
] = filtered_df.loc[
    (filtered_df['is_incorrect'] == True) & (filtered_df['is_incorrect_2'] == False),
    ['merged_transcription_2', 'is_incorrect_2']
].values


# Save processed data
column_mappings = {
    'filename': 'file_name',
    'text': 'original_transcription',
    'wav2vec2_transcription': 'whisper_transcription',
    'wav2vec_stress_txranscription': 'stressified_transcription',
    'merged_transcription_corrected': 'stressified_merged',
    'is_incorrect': 'is_incorrect'
}

final_df = filtered_df.rename(columns=column_mappings)[list(column_mappings.values())]
final_df = final_df.reset_index(drop=True)
final_df.to_csv('./cleaned/voa_stressed.csv')


filtered_df = pd.read_csv("./cleaned/voa_stressed.csv")

mask = (filtered_df['is_incorrect'] == True) & (filtered_df['is_incorrect_2'] == False)
filtered_df.loc[mask, 'wav2vec2_transcription'] = filtered_df.loc[mask, 'text'].values
filtered_df.loc[mask, 'merged_transcription_corrected'] = filtered_df.loc[mask, 'merged_transcription_corrected_2'].values

mask = (filtered_df['is_incorrect'] == False) | (filtered_df['is_incorrect_2'] == False)
filtered_df = filtered_df[['merged_transcription_corrected']]
filtered_df = filtered_df.rename(columns={'merged_transcription_corrected': 'labels'})

filtered_df['labels'] = filtered_df['labels'].str.strip()
filtered_df['text'] = filtered_df['labels'].apply(lambda x: x.replace('+', ''))

filtered_df.to_csv('voa_stressed_all_data.csv', index=False)

filtered_df = filtered_df[mask]
filtered_df.to_csv('voa_stressed_cleaned_data.csv', index=False)