import re
import pandas as pd
import os

from .accuracy import SentenceAccuracy

class WordDetectionError(Exception):
    """Custom exception for word detection errors."""
    pass

class SentenceMismatchError(Exception):
    """Custom exception for sentence mismatch errors."""
    pass

heteronyms_df = pd.read_csv(os.path.join("..", "benchmark", "data", "heteronyms", "heteronyms.csv"))

def is_word_heteronym(word: str):
    word = word.replace('', '').lower()
    word_set = set(heteronyms_df["Word"])
    return word in word_set

def evaluate_stress_word_level(correct_word: str, stressified_word: str, stress_mark: str='+'):
    """
    Evaluates if a word in the stressified sentence matches the correct word in terms of stress placement.

    Parameters:
    correct_word (str): The correctly stressified word containing accents marked with `stress_mark` (default is `+`).
    stressified_word (str): The word from the stressified sentence to be evaluated.
    stress_mark (str, optional): The symbol used to mark the stress (default is `+`).

    Returns:
    bool: True if the stressified word is correct according to the correct word, False otherwise.

    Examples:
    >>> evaluate_stress_word_level("За+вжди+", "За+вжди") # Case 1
    True
    >>> evaluate_stress_word_level("За+вжди+", "Завжди+") # Case 5
    True
    >>> evaluate_stress_word_level("За+вжди+", "За+вжди+") # Case 5
    True
    >>> evaluate_stress_word_level("За+вжди+", "Завжди") # Case 3
    False
    >>> evaluate_stress_word_level("За+вжди+", "За+вж+ди") # Case 4 all stress marks should be correct
    False
    >>> evaluate_stress_word_level("Кві+тка", "Квітка+") # Case 6
    False
    >>> evaluate_stress_word_level("так", "так") # Case 1
    True
    >>> evaluate_stress_word_level("так", "та+к") # Case 2 One-vowel word
    True
    >>> evaluate_stress_word_level("світе+", "світу+") # Case 6
    False
    """

    # 1: If both words are exactly the same, it's correct
    if correct_word == stressified_word:
        return True

    # 2. One vowel word: If the stressified word has one vowel, and it's marked as stressed - it's correct
    ukrainian_vowels_pattern = r'[АаЕеЄєИиІіЇїОоУуЮюЯя]'
    if len(stressified_word)-1==len(correct_word) and len(re.findall(ukrainian_vowels_pattern, stressified_word))==1:
        return True

    # 3: If stressified word do not have any stress marks - incorrect
    if stress_mark not in stressified_word:
        return False

    # 4: if length of words are the same, that's mean the stresses are different positions
    if len(correct_word) == len(stressified_word):
        return False

    # 5: If a word has several possible stress marks - let's check all variants
    for i in range(len(correct_word)):
        if correct_word[i] == stress_mark:
            correct_word_variant = correct_word[:i] + correct_word[i+1:]

            if correct_word_variant == stressified_word:
                return True

    # 6: Otherwise incorrect
    return False


def evaluate_stress_sentence_level(correct_sentence: str, stressified_sentence: str, stress_mark: str='+'):
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
    correct_words = re.findall(r'\S+', correct_sentence)
    stressified_words = re.findall(r'\S+', stressified_sentence)

    accuracy = SentenceAccuracy()
    accuracy.word_count = len(correct_words)

    if accuracy.word_count != len(stressified_words):
        raise SentenceMismatchError("The number of words in the sentences does not match.")

    is_heteronym = False

    for correct_word, stressified_word in zip(correct_words, stressified_words):

        # This block is executed only if a multi-vowel word in the benchmark has no stress mark,
        # indicating that labelers were unable to correctly identify the word's stress.
        plus_pattern = fr'[{stress_mark}]'
        ukrainian_vowels_pattern = r'[АаЕеЄєИиІіЇїОоУуЮюЯя]'
        if len(re.findall(plus_pattern, correct_word)) == 0 and len(re.findall(ukrainian_vowels_pattern, correct_word)) > 1:
            accuracy.word_count -=1
            continue

        if is_word_heteronym(correct_word.replace(stress_mark, '')):
            accuracy.heteronym_count += 1
            is_heteronym = True

        if evaluate_stress_word_level(correct_word, stressified_word, stress_mark):
            accuracy.correct_word_count += 1

            if is_heteronym:
                accuracy.correct_heteronym_count += 1
        is_heteronym = False

    if accuracy.word_count == 0:
        raise WordDetectionError("No words were detected in the sentence.")

    return accuracy
