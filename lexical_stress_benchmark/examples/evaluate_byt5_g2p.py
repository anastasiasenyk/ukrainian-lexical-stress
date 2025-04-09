import os
import re
import torch
from nemo.collections.tts.models.base import G2PModel

from lexical_stress_benchmark.benchmark import evaluate_stressification, shift_stress_marks


def load_g2p_model(model_name=None):
    """
    Load a G2P model from NeMo pretrained models or local path
    """
    if torch.cuda.is_available():
        device = [0]
        accelerator = "gpu"
    else:
        device = 1
        accelerator = "cpu"

    map_location = torch.device(
        "cuda:{}".format(device[0]) if accelerator == "gpu" else "cpu"
    )

    if os.path.exists(model_name):
        print('Loading pretrained G2P model')
        model = G2PModel.restore_from(model_name, map_location=map_location)
    else:
        raise ValueError(f"Model not found.")

    model = model.eval()
    return model


def text_to_phonemes(text, model):
    """
    Convert input text to phonemes using the G2P model's convert_graphemes_to_phonemes method
    but operating in memory instead of using files.
    """
    import json
    import tempfile
    from pathlib import Path

    # Create a temporary manifest file with the input text
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_input:
        input_data = [{"text_graphemes": text}]
        for item in input_data:
            temp_input.write(json.dumps(item) + "\n")
        input_path = temp_input.name

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_output:
        output_path = temp_output.name

    try:
        # Run the conversion
        with torch.no_grad():
            model.convert_graphemes_to_phonemes(
                manifest_filepath=input_path,
                output_manifest_filepath=output_path,
                grapheme_field="text_graphemes",
                pred_field="pred_text",
            )

        # Read the output
        with open(output_path, "r") as f:
            output_data = [json.loads(line) for line in f]

        # Return the phonetic transcription
        if output_data and "pred_text" in output_data[0]:
            return output_data[0]["pred_text"]
        else:
            return "No phonetic transcription generated"

    finally:
        # Clean up temporary files
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
    return phonemes


UKRAINIAN_LETTERS = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
UKRAINIAN_LETTERS += UKRAINIAN_LETTERS.upper()

def custom_stressify(text):
    text = text.strip()

    match = re.search(f"[{UKRAINIAN_LETTERS}]", text)
    if match:
        i = match.start()
        prefix = text[:i]
        word = text[i:]
        result = text_to_phonemes(word, model)
        result = shift_stress_marks(result)
        return prefix + result
    else:
        return text


if __name__ == "__main__":
    model_path = "../../accentor_nemo/2025-04-07_13-11-55/checkpoints/T5G2P.nemo"
    model = load_g2p_model(model_path)

    accuracies = evaluate_stressification(custom_stressify, stress_mark='+', raise_on_sent_mismatch=False)
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy = accuracies.values()

    print('Byt5 G2P results:')
    print(f"{'Sentence Accuracy:':30} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':30} {word_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':30} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':30} {unambiguous_accuracy * 100:.2f}%")
