import os

from accentor_model.predict_word_stress import Stressifier
from huggingface_hub import hf_hub_download

from lexical_stress_benchmark.benchmark import evaluate_stressification

path_to_nemo_model = "../byt5_g2p/experiment/checkpoints/T5G2P.nemo"

if not os.path.exists(path_to_nemo_model):
    path_to_nemo_model = hf_hub_download(
        repo_id="mouseyy/stressifier-byt5-g2p-model",
        filename="T5G2P.nemo",
        token="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",
    )

stressifier = Stressifier(path_to_nemo_model)


def custom_stressify(text):
    return stressifier.stressify(text)


if __name__ == "__main__":
    accuracies = evaluate_stressification(custom_stressify)
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy, macro_average_f1_across_heteronyms = (
        accuracies.values()
    )

    print("Byt5 G2P results:")
    print(f"{'Sentence Accuracy:':40} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':40} {word_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':40} {unambiguous_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':40} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Macro-Average F1 score (Heteronyms)):':40} {macro_average_f1_across_heteronyms * 100:.2f}%")
