from benchmark.benchmark import evaluate_stressification
from ukrainian_word_stress import Stressifier, OnAmbiguity  # pip install git+https://github.com/egorsmkv/ukrainian-accentor.git


stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)

def custom_stressify(text):
    return stressify(text)

if __name__ == '__main__':
    sentence_accuracy, word_accuracy, heteronym_accuracy = evaluate_stressification(custom_stressify, stress_mark='+')

    print('Ukrainian Word Stress results:')
    print(f"Sentence Accuracy: {sentence_accuracy:.2f}%")
    print(f"Word Accuracy: {word_accuracy:.2f}%")
    print(f"Heteronym Accuracy: {heteronym_accuracy:.2f}%")


    # Ukrainian Word Stress results:
    # Sentence Accuracy: 0.10%
    # Word Accuracy: 0.72%
    # Heteronym Accuracy: 0.64%