from lexical_stress_benchmark import evaluate_stressification, shift_stress_marks
import ukrainian_accentor as accentor # pip install ukrainian_word_stress


def custom_stressify(text):
    result = accentor.process(text, mode='plus')
    return shift_stress_marks(result)


if __name__ == '__main__':
    accuracies = evaluate_stressification(custom_stressify, stress_mark='+')
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy = accuracies.values()

    print('Ukrainian Accentor results:')

    print(f"{'Sentence Accuracy:':30} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':30} {word_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':30} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':30} {unambiguous_accuracy * 100:.2f}%")

    # 09.04.2025
    # Ukrainian Accentor results:
    #
    # Sentence Accuracy:             16.57%
    # Word Accuracy:                 73.16%
    # Heteronym Accuracy:            41.63%
    # Unambiguous Words Accuracy:    78.68%