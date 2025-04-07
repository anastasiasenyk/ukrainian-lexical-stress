import os
import torch
import gradio as gr
import lightning.pytorch as pl
from nemo.collections.tts.models.base import G2PModel


def load_g2p_model(model_name=None):
    """
    Load a G2P model from NeMo pretrained models or local path
    """
    # Setup GPU if available
    if torch.cuda.is_available():
        device = [0]
        accelerator = "gpu"
    else:
        device = 1
        accelerator = "cpu"

    map_location = torch.device(
        "cuda:{}".format(device[0]) if accelerator == "gpu" else "cpu"
    )
    trainer = pl.Trainer(
        devices=device,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
    )

    # Load model
    if model_name is None:
        # Use the first available model if none specified
        available_models = G2PModel.list_available_models()
        if not available_models:
            raise ValueError("No pretrained G2P models available")
        model_name = available_models[0]
        print(f"Using default model: {model_name}")

    if os.path.exists(model_name):
        model = G2PModel.restore_from(model_name, map_location=map_location)
    elif model_name in G2PModel.get_available_model_names():
        model = G2PModel.from_pretrained(model_name, map_location=map_location)
    else:
        raise ValueError(
            f"Model not found. Choose from {G2PModel.list_available_models()} or provide a valid path"
        )

    model._cfg.max_source_len = 512
    model.set_trainer(trainer)
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
                batch_size=32,
                num_workers=0,
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


# Load the model at startup

model_path = "nemo_experiments/T5G2P/2025-04-07_13-11-55/checkpoints/T5G2P.nemo"
model = load_g2p_model(model_path)


def process_text(input_text):
    """
    Process input text and return phonetic transcription
    """
    if not input_text.strip():
        return "Please enter some text."

    try:
        print(f"Processing input text: {input_text}")
        phonetic_result = text_to_phonemes(input_text, model)
        print(f"Phonetic result: {phonetic_result}")

        if not phonetic_result:
            return "Could not generate phonetic transcription. Try a different text."

        return phonetic_result
    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing text: {str(e)}\nPlease check the console for more details."


# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text to convert to phonemes..."),
    outputs=gr.Textbox(label="Phonetic Transcription"),
    title="Grapheme to Phoneme Converter",
    description="Convert text to phonetic representation using NeMo's G2P model",
    examples=[
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "I pronounce this correctly",
    ],
)

if __name__ == "__main__":
    demo.launch()
