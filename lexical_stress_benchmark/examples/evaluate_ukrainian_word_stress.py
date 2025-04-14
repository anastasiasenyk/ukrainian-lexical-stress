from ukrainian_word_stress import OnAmbiguity, Stressifier

from lexical_stress_benchmark import evaluate_stressification

stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)


def custom_stressify(text):
    return stressify(text)


if __name__ == "__main__":
    accuracies = evaluate_stressification(custom_stressify)
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy, macro_average_f1_across_heteronyms = (
        accuracies.values()
    )

    print("Ukrainian Word Stress results:")

    print(f"{'Sentence Accuracy:':40} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':40} {word_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':40} {unambiguous_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':40} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Macro-Average F1 score (Heteronyms)):':40} {macro_average_f1_across_heteronyms * 100:.2f}%")
