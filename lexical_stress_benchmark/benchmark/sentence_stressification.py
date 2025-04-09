import re
import os
import pandas as pd
import marisa_trie
from typing import List

from lexical_stress_benchmark.benchmark.accuracy import SentenceAccuracy

class WordDetectionError(Exception):
    """Custom exception for word detection errors."""
    pass

class SentenceMismatchError(Exception):
    """Custom exception for sentence mismatch errors."""
    pass

HETERONYMS = set(pd.read_csv(os.path.join("..", "data", "heteronyms_list.csv"))["Heteronym"])
VOCABULARY = marisa_trie.BytesTrie().load(os.path.join("..", "data", 'stress.trie'))

def is_word_heteronym(word: str):
    word = word.replace('+', '').lower()
    return word in HETERONYMS


def is_word_without_ambiguity(word: str):
    for word in [word.lower(), word.upper()]:
        if word in VOCABULARY:
            if len(VOCABULARY[word][0]) <= 1:
                return True
    return False


def evaluate_stress_word_level(
    correct: str,
    candidate: str,
    stress_symbol: str = '+'
) -> bool:
    """
    Evaluates whether the stress pattern in `candidate` matches the correct pattern in `correct`.

    Args:
        correct (str): The reference word with correct stress positions marked.
        candidate (str): The stressified word to validate.
        stress_symbol (str): The symbol used to mark stress (default is '+').

    Returns:
        bool: True if `candidate` matches the stress pattern of `correct`, False otherwise.
    """

    def get_stressed_char_positions(word: str, symbol: str) -> List[int]:
        """
        Returns a list of character positions where stress is applied.
        The position is relative to characters (excluding stress symbols).
        """
        positions = []
        char_index = 0
        for char in word:
            if char == symbol:
                positions.append(char_index)
            else:
                char_index += 1
        return positions

    def remove_stress_marks(word: str, symbol: str) -> str:
        """Removes all stress symbols from the word."""
        return word.replace(symbol, '')

    # Rule 1: Base form without stress symbols must match
    base_correct = remove_stress_marks(correct, stress_symbol)
    base_candidate = remove_stress_marks(candidate, stress_symbol)
    if base_correct != base_candidate:
        return False

    # Rule 2: Exact match with stress marks is valid
    if correct == candidate:
        return True

    # Rule 3: Words with only one vowel are always considered correct
    vowel_pattern = r'[АаЕеЄєИиІіЇїОоУуЮюЯя]'
    if len(re.findall(vowel_pattern, candidate)) == 1:
        return True

    # Rule 4: If candidate has no stress at all, it's incorrect
    if stress_symbol not in candidate:
        return False

    # Rule 5: Compare stress positions — all stresses in candidate must exist in correct
    correct_stresses = get_stressed_char_positions(correct, stress_symbol)
    candidate_stresses = get_stressed_char_positions(candidate, stress_symbol)

    return set(candidate_stresses).issubset(correct_stresses)


def evaluate_stress_sentence_level(correct_sentence: str, stressified_sentence: str, stress_mark: str='+', raise_on_mismatch: bool=True):
    """
    Evaluates the stressification of a sentence by comparing each word.

    This function calculates two metrics:
    - Word-level accuracy: The percentage of correctly stressified words.
    - Sentence-level accuracy: Whether the entire sentence has been stressified correctly (same as all words are correct).

    Parameters:
    correct_sentence (str): The correctly stressified sentence where accents are marked with 'stress_mark' (default is '+').
    stressified_sentence (str): The stressified sentence to be evaluated.
    stress_mark (str, optional): The symbol used to mark the stress (default is '+').

    Returns:
    SentenceAccuracy: An instance of the `SentenceAccuracy` class containing the following attributes:
        - word_count (int): The total number of words in the sentence.
        - correct_word_count (int): The number of correctly stressified words.
        - heteronym_count (int): The number of heteronym words in the sentence.
        - correct_heteronym_count (int): The number of correctly stressified heteronym words.
    """
    correct_sentence = correct_sentence.strip()
    stressified_sentence = stressified_sentence.strip()

    correct_words = re.findall(r'\S+', correct_sentence)
    stressified_words = re.findall(r'\S+', stressified_sentence)

    accuracy = SentenceAccuracy()
    accuracy.total_words = len(correct_words)

    if correct_sentence.replace('+', '') != stressified_sentence.replace('+', '') or len(correct_words) != len(stressified_words):
        if raise_on_mismatch:
            raise SentenceMismatchError("The number of words in the sentences does not match.")
        return accuracy

    is_heteronym = False
    is_unambiguity = False

    for correct_word, stressified_word in zip(correct_words, stressified_words):

        plus_pattern = fr'[{stress_mark}]'
        ukrainian_vowels_pattern = r'[АаЕеЄєИиІіЇїОоУуЮюЯя]'
        if len(re.findall(plus_pattern, correct_word)) == 0 and len(re.findall(ukrainian_vowels_pattern, correct_word)) > 1:
            # This block is executed only if a multi-vowel word in the benchmark has no stress mark,
            # indicating that labelers were unable to correctly identify the word's stress.
            accuracy.total_words -= 1
            continue

        if len(re.findall(ukrainian_vowels_pattern, correct_word)) <= 1: # skip words with one vowel
            accuracy.total_words -= 1
            continue

        if is_word_heteronym(correct_word.replace(stress_mark, '')):
            accuracy.total_heteronyms += 1
            is_heteronym = True

        if is_word_without_ambiguity(correct_word.replace(stress_mark, '')):
            accuracy.total_unambiguous_words += 1
            is_unambiguity = True

        if evaluate_stress_word_level(correct_word, stressified_word, stress_mark):
            accuracy.correctly_stressified_words += 1

            if is_heteronym:
                accuracy.correctly_stressified_heteronyms += 1

            if is_unambiguity:
                accuracy.correctly_stressified_unambiguous += 1

        is_heteronym = False
        is_unambiguity = False

    if accuracy.total_words == 0:
        raise WordDetectionError("No words were detected in the sentence.")

    return accuracy
