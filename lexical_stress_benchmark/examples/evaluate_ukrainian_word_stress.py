from ukrainian_word_stress import OnAmbiguity, Stressifier

from lexical_stress_benchmark import evaluate_stressification

stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)


def custom_stressify(text):
    return stressify(text)


if __name__ == "__main__":
    accuracies = evaluate_stressification(custom_stressify)
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy = accuracies.values()

    print("Ukrainian Word Stress results:")

    print(f"{'Sentence Accuracy:':30} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':30} {word_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':30} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':30} {unambiguous_accuracy * 100:.2f}%")
