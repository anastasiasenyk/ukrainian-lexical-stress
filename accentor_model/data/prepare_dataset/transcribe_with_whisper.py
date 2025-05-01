import os
import pandas as pd
import whisper
from tqdm import tqdm


def find_and_delete_wav_files(input_dir, df):
    """
    Recursively finds all .wav files in the directory matching the pattern ./*/*/*.wav,
    checks if the filename exists in the DataFrame's 'file' column,
    and deletes files that do not match.
    """
    # Get all .wav files in the format ./*/*/*.wav
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav") and len(root.split(os.sep)) >= 3:
                wav_files.append(os.path.join(root, file))
    wav_filenames = {os.path.basename(path) for path in wav_files}

    df_filenames = {os.path.basename(path) for path in df['file'].astype(str)}

    non_matching_files = wav_filenames - df_filenames

    for file_path in wav_files:
        if os.path.basename(file_path) in non_matching_files:
            os.remove(file_path)
    return non_matching_files


def transcribe_audio(file_path):
    """
    Transcribes an audio file to Ukrainian using Whisper on a specified GPU.
    """
    result = model.transcribe(file_path, language="uk")
    return result["text"]


def process_and_transcribe(input_dir, df, batch_size=10):
    """
    Processes .wav files, transcribes them to Ukrainian, and saves the results in a new CSV file.
    """
    df['file_name'] = df['file'].apply(lambda x: x.split("/")[-1])
    known_files = set(df['file_name'].values)

    transcribed_data = []

    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav") and len(root.split(os.sep)) >= 3:
                wav_files.append(os.path.join(root, file))

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav") and len(root.split(os.sep)) >= 3:
                wav_files.append(os.path.join(root, file))

    for file_path in tqdm(wav_files, desc="Processing and Transcribing", unit="file"):
        file_name = os.path.basename(file_path)

        if file_name in known_files:
            file_data = df[df['file_name'] == file_name].iloc[0].to_dict()

            transcription = transcribe_audio(file_path)
            file_data["transcription_ukrainian"] = transcription

            transcribed_data.append(file_data)

        if len(transcribed_data) >= batch_size:
            batch_df = pd.DataFrame(transcribed_data)
            batch_df.to_csv("voa_transcribed.csv", mode='a', header=False, index=False)
            transcribed_data = []

    if transcribed_data:
        batch_df = pd.DataFrame(transcribed_data)
        batch_df.to_csv("voa_transcribed.csv", mode='a', header=False, index=False)


model = whisper.load_model("large", device="cuda")

# Example usage:
data_dir = "."
df = pd.read_csv("voa_clean.csv")
find_and_delete_wav_files(data_dir, df)
process_and_transcribe(data_dir, df, batch_size=10)