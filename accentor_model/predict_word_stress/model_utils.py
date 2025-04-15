import json
import os
import tempfile
from pathlib import Path

import torch
from nemo.collections.tts.models.base import G2PModel


def load_g2p_model(model_path: str = None):
    """
    Loads a G2P model from a file path.

    Args:
        model_path (str, optional): Path to the .nemo G2P model file.

    Returns:
        G2PModel: The loaded and ready-to-use model.

    Raises:
        ValueError: If the model file does not exist.
    """
    if torch.cuda.is_available():
        device = [0]
        accelerator = "gpu"
    else:
        device = 1
        accelerator = "cpu"

    map_location = torch.device("cuda:{}".format(device[0]) if accelerator == "gpu" else "cpu")

    if os.path.exists(model_path):
        model = G2PModel.restore_from(model_path, map_location=map_location)
    else:
        raise ValueError(f"Model not found at {model_path}")

    return model.eval()


def text_to_phonemes(text: str, model: G2PModel) -> str:
    """
    Converts graphemes to phonemes using a G2P model.

    Args:
        text (str): The cleaned input string.
        model: The G2P model object.

    Returns:
        str: The phonetic transcription with stress marks.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_input:
        temp_input.write(json.dumps({"text_graphemes": text}) + "\n")
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_output:
        output_path = temp_output.name

    try:
        with torch.no_grad():
            model.convert_graphemes_to_phonemes(
                manifest_filepath=input_path,
                output_manifest_filepath=output_path,
                grapheme_field="text_graphemes",
                pred_field="pred_text",
            )

        with open(output_path, "r") as f:
            output_data = [json.loads(line) for line in f]

        if output_data and "pred_text" in output_data[0]:
            return output_data[0]["pred_text"]
        else:
            return ""
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
