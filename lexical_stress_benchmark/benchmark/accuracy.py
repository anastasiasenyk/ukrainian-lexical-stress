from collections import defaultdict

import numpy as np


class SentenceAccuracy:
    """
    Represents accuracy metrics for a single sentence's stressification.

    Attributes:
        total_words (int): Total words in the sentence.
        correctly_stressified_words (int): Words stressified correctly.
        total_heteronyms (int): Total heteronyms in the sentence.
        correctly_stressified_heteronyms (int): Heteronyms stressified correctly.
        total_unambiguous_words (int): Known words with no ambiguity.
        correctly_stressified_unambiguous (int): Correctly stressified known, unambiguous words.
        heteronyms_dictionary (defaultdict): Records predictions for heteronyms.
            Structure: plain_word -> gold_variant -> list of predicted_variants
    """

    def __init__(self) -> None:
        self.total_words: int = 0
        self.correctly_stressified_words: int = 0
        self.total_heteronyms: int = 0
        self.correctly_stressified_heteronyms: int = 0
        self.total_unambiguous_words: int = 0
        self.correctly_stressified_unambiguous: int = 0

        # Structure: plain_word -> label_variant -> list of predicted variants
        self.heteronyms_dictionary: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def is_sentence_correct(self) -> bool:
        """Checks if all words in the sentence were stressified correctly."""
        return self.total_words == self.correctly_stressified_words

    def __eq__(self, other: object) -> bool:
        """Checks equality with another SentenceAccuracy object."""
        if not isinstance(other, SentenceAccuracy):
            return False
        return (
            self.total_words == other.total_words
            and self.correctly_stressified_words == other.correctly_stressified_words
            and self.total_heteronyms == other.total_heteronyms
            and self.correctly_stressified_heteronyms == other.correctly_stressified_heteronyms
            and self.total_unambiguous_words == other.total_unambiguous_words
            and self.correctly_stressified_unambiguous == other.correctly_stressified_unambiguous
        )

    def __repr__(self) -> str:
        return (
            f"SentenceAccuracy(total_words={self.total_words}, "
            f"correctly_stressified_words={self.correctly_stressified_words}, "
            f"total_heteronyms={self.total_heteronyms}, "
            f"correctly_stressified_heteronyms={self.correctly_stressified_heteronyms}, "
            f"total_unambiguous_words={self.total_unambiguous_words}, "
            f"correctly_stressified_unambiguous={self.correctly_stressified_unambiguous})"
        )


class DatasetAccuracy:
    """
    Aggregates and computes dataset-level stressification accuracy metrics.

    Attributes:
        total_words (int): Total words in the dataset.
        correctly_stressified_words (int): Correctly stressified words in the dataset.
        total_heteronyms (int): Total heteronyms in the dataset.
        correctly_stressified_heteronyms (int): Correctly stressified heteronyms.
        total_unambiguous_words (int): Total unambiguous words in the dataset.
        correctly_stressified_unambiguous (int): Correctly stressified unambiguous words.
        fully_correct_sentences (int): Sentences where all words are correct.
        heteronyms_dictionary (defaultdict): Records predictions for heteronyms.
            Structure: plain_word -> gold_variant -> list of predicted_variants
    """

    def __init__(self) -> None:
        self.total_words: int = 0
        self.correctly_stressified_words: int = 0
        self.total_heteronyms: int = 0
        self.correctly_stressified_heteronyms: int = 0
        self.total_unambiguous_words: int = 0
        self.correctly_stressified_unambiguous: int = 0
        self.fully_correct_sentences: int = 0

        # Structure: plain_word -> label_variant -> list of predicted variants
        self.heteronyms_dictionary: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def update_with_sentence(self, sentence_accuracy: SentenceAccuracy) -> None:
        """Accumulates accuracy metrics from a sentence."""
        self.total_words += sentence_accuracy.total_words
        self.correctly_stressified_words += sentence_accuracy.correctly_stressified_words
        self.total_heteronyms += sentence_accuracy.total_heteronyms
        self.correctly_stressified_heteronyms += sentence_accuracy.correctly_stressified_heteronyms
        self.total_unambiguous_words += sentence_accuracy.total_unambiguous_words
        self.correctly_stressified_unambiguous += sentence_accuracy.correctly_stressified_unambiguous
        self.fully_correct_sentences += int(sentence_accuracy.is_sentence_correct())

        for key, subdict in sentence_accuracy.heteronyms_dictionary.items():
            for subkey, values in subdict.items():
                self.heteronyms_dictionary[key][subkey].extend(values)

    def compute_averages(self, total_sentences: int) -> dict[str, float]:
        """
        Computes average accuracy metrics across the dataset.

        Returns:
            Dictionary with sentence, word, heteronym, and unambiguous word accuracies.
        """
        macro_f1s = []

        for plain_word, label_variants in self.heteronyms_dictionary.items():
            # Calculate only if there are at least two variants of the heteronym in the dataset
            if len(label_variants) < 2:
                continue

            f1s = []
            for label_variant, predictions in label_variants.items():
                TP = sum(pred == label_variant for pred in predictions)
                FN = sum(pred != label_variant for pred in predictions)
                FP = sum(
                    pred == label_variant
                    for other_variant, other_preds in label_variants.items()
                    if other_variant != label_variant
                    for pred in other_preds
                )

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                f1s.append(f1)

            macro_f1s.append(np.mean(f1s))

        return {
            "sentence_accuracy": self.fully_correct_sentences / total_sentences if total_sentences else 0.0,
            "word_accuracy": self.correctly_stressified_words / self.total_words if self.total_words else 0.0,
            "heteronym_accuracy": (
                self.correctly_stressified_heteronyms / self.total_heteronyms if self.total_heteronyms else 0.0
            ),
            "unambiguous_accuracy": (
                self.correctly_stressified_unambiguous / self.total_unambiguous_words
                if self.total_unambiguous_words
                else 0.0
            ),
            "macro_average_f1_across_heteronyms": np.mean(macro_f1s),
        }
