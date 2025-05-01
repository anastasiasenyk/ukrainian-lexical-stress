import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCTC
import glob
from tqdm import tqdm

model_checkpoint = "mouseyy/uk_wav2vec2_with_stress_mark"

config = AutoConfig.from_pretrained(model_checkpoint)
tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None

tokenizer = AutoTokenizer.from_pretrained(
  model_checkpoint,
  config=config,
  tokenizer_type=tokenizer_type,
  unk_token="[UNK]",
  pad_token="[PAD]",
  word_delimiter_token="|",
)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = AutoModelForCTC.from_pretrained(
    model_checkpoint,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def prepare_audio(file_path):
    file_path = file_path.split('/')[-1]

    pattern = f'./*/*/{file_path}'
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f'No audio files found with pattern {pattern}')

    file_path = matching_files[0]
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze()
    return waveform, sample_rate


def transcribe_files(df):
    # Initialize a list to hold the transcriptions
    transcriptions = []

    # Iterate through each row of the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing files"):
        file_path = row['file']

        waveform, sample_rate = prepare_audio(file_path)

        input_values = processor(waveform, sampling_rate=sample_rate).input_values
        input_values = torch.tensor(input_values).to(device)

        # Get the logits from the model
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the predictions
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]
        transcriptions.append(transcription)

    # Add the transcriptions to the DataFrame
    df['wav2vec_stress_transcription'] = transcriptions
    return df


print("Train set: Start")

df = pd.read_csv("./voa_transcribed_236k_rows.csv")
df_with_transcriptions = transcribe_files(df)
df_with_transcriptions.to_csv("../raw/voa_transcribed_236k_rows_stressed.csv", index=False)

print("Train set: Done")