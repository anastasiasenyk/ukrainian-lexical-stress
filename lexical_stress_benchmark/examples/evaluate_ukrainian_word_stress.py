from lexical_stress_benchmark import evaluate_stressification
from ukrainian_word_stress import OnAmbiguity, Stressifier

stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)


def custom_stressify(text):
    return stressify(text)


if __name__ == "__main__":
    accuracies = evaluate_stressification(custom_stressify, stress_mark="+")
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy = accuracies.values()

    print("Ukrainian Word Stress results:")

    print(f"{'Sentence Accuracy:':30} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':30} {word_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':30} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':30} {unambiguous_accuracy * 100:.2f}%")

    # 09.04.2025
    # Ukrainian Word Stress results:
    #
    # OnAmbiguity.First:
    # Sentence Accuracy:             41.62%
    # Word Accuracy:                 88.65%
    # Heteronym Accuracy:            64.35%
    # Unambiguous Words Accuracy:    98.60%
    #
    # OnAmbiguity.Skip:
    # Sentence Accuracy:             32.55%
    # Word Accuracy:                 85.92%
    # Heteronym Accuracy:            42.28%
    # Unambiguous Words Accuracy:    98.58%
    #
    # OnAmbiguity.All:
    # Ukrainian Word Stress results:
    # Sentence Accuracy:             33.63%
    # Word Accuracy:                 86.12%
    # Heteronym Accuracy:            44.89%
    # Unambiguous Words Accuracy:    98.58%
