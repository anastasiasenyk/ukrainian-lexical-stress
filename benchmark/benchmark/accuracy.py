
class SentenceAccuracy:
    """
    A class to hold and compute sentence-level accuracy metrics.

    This class tracks the accuracy of stressification for a single sentence,
    focusing on word-level accuracy and heteronym-specific accuracy.

    Attributes:
        word_count (int): The total number of words in the sentence.
        correct_word_count (int): The number of correctly stressified words.
        heteronym_count (int): The number of heteronym words in the sentence.
        correct_heteronym_count (int): The number of correctly stressified heteronym words.
    """
    def __init__(self):
        # to calculate word accuracy
        self.word_count = 0
        self.correct_word_count = 0
        # to calculate heteronym accuracy
        self.heteronym_count = 0
        self.correct_heteronym_count = 0

    def get_sentence_accuracy(self):
        """
        Determines if the sentence is fully accurate in terms of word stressification.
        """
        return 1 if self.correct_word_count == self.word_count else 0

    def __eq__(self, other):
        if not isinstance(other, SentenceAccuracy):
            return False
        return (
                self.word_count == other.word_count and
                self.correct_word_count == other.correct_word_count and
                self.heteronym_count == other.heteronym_count and
                self.correct_heteronym_count == other.correct_heteronym_count
        )


class AccuracyMetrics:
    """
    A class to store and calculate dataset-level accuracy metrics.

    This class aggregates accuracy metrics from individual sentences and computes
    the overall accuracy at the dataset level. The metrics include word-level accuracy,
    heteronym accuracy, and sentence-level accuracy.

    Attributes:
        word_count (int): The total number of words in the dataset.
        correct_word_count (int): The total number of correctly stressified words in the dataset.
        heteronym_count (int): The total number of heteronym words in the dataset.
        correct_heteronym_count (int): The total number of correctly stressified heteronym words in the dataset.
        correct_sentence_count (int): The total number of sentences where all words were stressified correctly.
    """
    def __init__(self):
        # to calculate word accuracy
        self.word_count = 0
        self.correct_word_count = 0
        # to calculate heteronym accuracy
        self.heteronym_count = 0
        self.correct_heteronym_count = 0
        # to calculate sentence accuracy
        self.correct_sentence_count = 0

    def accumulate(self, accuracy: SentenceAccuracy):
        """
        Accumulates the accuracy metrics from a single sentence into the dataset-level metrics.
        """
        self.word_count += accuracy.word_count
        self.correct_word_count += accuracy.correct_word_count
        self.heteronym_count += accuracy.heteronym_count
        self.correct_heteronym_count += accuracy.correct_heteronym_count
        self.correct_sentence_count += accuracy.get_sentence_accuracy()

    def average_accuracies(self, total_sentences):
        """
        Calculates the average accuracies for word stressification, heteronym stressification,
        and sentence stressification at the dataset level.
        """
        word_accuracy = self.correct_word_count / self.word_count if self.word_count else 0
        heteronym_accuracy = self.correct_heteronym_count / self.heteronym_count if self.heteronym_count else 0
        sentence_accuracy = self.correct_sentence_count / total_sentences if total_sentences else 0
        return sentence_accuracy, word_accuracy, heteronym_accuracy


