import os
import re
from typing import List

import marisa_trie
import pandas as pd

from lexical_stress_benchmark.benchmark.accuracy import SentenceAccuracy


class WordDetectionError(Exception):
    """Custom exception for word detection errors."""

    pass


class SentenceMismatchError(Exception):
    """Custom exception for sentence mismatch errors."""

    pass


HETERONYMS = set(pd.read_csv(os.path.join("..", "data", "heteronyms_list.csv"))["Heteronym"])
VOCABULARY = marisa_trie.BytesTrie().load(os.path.join("..", "data", "stress.trie"))


def is_word_heteronym(word: str):
    return word in HETERONYMS


def is_word_unambiguous(word: str):
    for word in [word.lower(), word.upper()]:
        if word in VOCABULARY:
            if len(VOCABULARY[word][0]) <= 1:
                return True
    return False


def evaluate_stress_word_level(correct: str, candidate: str, stress_symbol: str = "+") -> bool:
    """
    Evaluates whether the stress pattern in `candidate` matches the correct pattern in `correct`.

    Args:
        correct (str): The reference word with correct stress positions marked.
        candidate (str): The stressified word to validate.
        stress_symbol (str, optional): The symbol used to mark stress (default is '+').

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
        return word.replace(symbol, "")

    # Rule 1: Base form without stress symbols must match
    base_correct = remove_stress_marks(correct, stress_symbol)
    base_candidate = remove_stress_marks(candidate, stress_symbol)
    if base_correct != base_candidate:
        return False

    # Rule 2: Exact match with stress marks is valid
    if correct == candidate:
        return True

    # Rule 3: Words with only one vowel are always considered correct
    vowel_pattern = r"[АаЕеЄєИиІіЇїОоУуЮюЯя]"
    if len(re.findall(vowel_pattern, candidate)) == 1:
        return True

    # Rule 4: If candidate has no stress at all, it's incorrect
    if stress_symbol not in candidate:
        return False

    # Rule 5: Compare stress positions — all stresses in candidate must exist in correct
    correct_stresses = get_stressed_char_positions(correct, stress_symbol)
    candidate_stresses = get_stressed_char_positions(candidate, stress_symbol)

    return set(candidate_stresses).issubset(correct_stresses)


def evaluate_stress_sentence_level(
    correct_sentence: str,
    candidate_sentence: str,
    stress_mark: str = "+",
    raise_on_mismatch: bool = True,
    ignore_mismatch: bool = False,
) -> SentenceAccuracy:
    """
    Evaluates the stressification of a sentence by comparing each word.

    Args:
        correct_sentence (str): The correctly stressified sentence where accents are marked with 'stress_mark'.
        candidate_sentence (str): The candidate sentence to be evaluated for stressification.
        stress_mark (str, optional): The symbol used to mark the stress (default is '+').
        raise_on_mismatch (bool, optional): If True, raises an error if the sentences have mismatched structure.
        ignore_mismatch (bool, optional): If True, returns None on mismatch instead of raising an error or proceeding with evaluation.

    Returns:
        SentenceAccuracy: An instance of the `SentenceAccuracy` class
    """
    correct_sentence = correct_sentence.strip()
    candidate_sentence = candidate_sentence.strip()

    correct_words = re.findall(r"\S+", correct_sentence)
    candidate_words = re.findall(r"\S+", candidate_sentence)

    accuracy = SentenceAccuracy()
    accuracy.total_words = len(correct_words)

    if correct_sentence.replace(stress_mark, "") != candidate_sentence.replace(stress_mark, "") or len(
        correct_words
    ) != len(candidate_words):
        if raise_on_mismatch:
            raise SentenceMismatchError("The number of words in the sentences does not match.")
        if ignore_mismatch:
            return None
        candidate_words = re.findall(r"\S+", candidate_sentence.replace(stress_mark, ""))

    is_heteronym = False
    is_unambiguous = False

    for correct_word, candidate_word in zip(correct_words, candidate_words):
        plain_word = correct_word.lower().replace(stress_mark, "")
        plus_pattern = rf"[{stress_mark}]"
        ukrainian_vowels_pattern = r"[АаЕеЄєИиІіЇїОоУуЮюЯя]"
        if (
            len(re.findall(plus_pattern, correct_word)) == 0
            and len(re.findall(ukrainian_vowels_pattern, correct_word)) > 1
        ):
            # This block is executed only if a multi-vowel word in the benchmark has no stress mark,
            # indicating that labelers were unable to correctly identify the word's stress.
            accuracy.total_words -= 1
            continue

        if len(re.findall(ukrainian_vowels_pattern, correct_word)) <= 1:  # skip words with <= 1 vowel
            accuracy.total_words -= 1
            continue

        if is_word_heteronym(plain_word):
            accuracy.total_heteronyms += 1
            is_heteronym = True
            accuracy.heteronyms_dictionary[plain_word][correct_word.lower()].append(candidate_word.lower())

        if is_word_unambiguous(plain_word):
            accuracy.total_unambiguous_words += 1
            is_unambiguous = True

        if evaluate_stress_word_level(correct_word, candidate_word, stress_mark):
            accuracy.correctly_stressified_words += 1

            if is_heteronym:
                accuracy.correctly_stressified_heteronyms += 1

            if is_unambiguous:
                accuracy.correctly_stressified_unambiguous += 1

        is_heteronym = False
        is_unambiguous = False

    if accuracy.total_words == 0:
        raise WordDetectionError("No words were detected in the sentence.")

    return accuracy
