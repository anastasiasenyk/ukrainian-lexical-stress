from benchmark.benchmark import evaluate_stressification
from ukrainian_word_stress import Stressifier, OnAmbiguity  # pip install git+https://github.com/egorsmkv/ukrainian-accentor.git


stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)

def custom_stressify(text):
    return stressify(text)

if __name__ == '__main__':
    sentence_accuracy, word_accuracy, heteronym_accuracy = evaluate_stressification(custom_stressify, stress_mark='+')

    print("Sentence Accuracy: ", sentence_accuracy)
    print("Word Accuracy: ", word_accuracy)
    print("Heteronym Accuracy: ", heteronym_accuracy)