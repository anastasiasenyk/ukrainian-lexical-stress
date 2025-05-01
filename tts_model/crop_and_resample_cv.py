import os
import torch
import torchaudio
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# Load the Silero VAD model
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=True)
(get_speech_timestamps, _, read_audio, *_) = utils


input_directory = './cv-corpus-19-1.0-2024-09-13/uk/clips'
output_directory = './cv-corpus-19-1.0-2024-09-13/wavs'


if not os.path.exists(output_directory):
    os.makedirs(output_directory)

train_df_1 = pd.read_csv('./prepared/temp/processed_train_1.csv')
train_df_2 = pd.read_csv('./prepared/temp/processed_train_2.csv')
test_df = pd.read_csv('./prepared/temp/processed_test.csv')
dev_df = pd.read_csv('./prepared/temp/processed_dev.csv')

metadata_df = pd.concat([train_df_1, train_df_2, test_df, dev_df], ignore_index=True)
all_filenames = metadata_df['path'].tolist()

def detect_speech_silero(wav_path, model, sample_rate=16000):
    wav = read_audio(wav_path, sampling_rate=sample_rate)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
    cropped_wav = torch.cat([wav[timestamp['start']:timestamp['end']] for timestamp in speech_timestamps])
    return cropped_wav


# Iterate through MP3 files in the input directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith(".mp3") and filename in all_filenames:
        # Load MP3 file using pydub
        mp3_path = os.path.join(input_directory, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.wav"
        output_path = os.path.join(output_directory, output_filename)

        try:
            audio = AudioSegment.from_mp3(mp3_path)

            # Convert to 16 kHz (Silero operates on 16 kHz)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            temp_wav_path = os.path.join(output_directory, os.path.splitext(filename)[0] + "_temp.wav")
            audio.export(temp_wav_path, format="wav")
            cropped_audio_16k = detect_speech_silero(temp_wav_path, silero_model, sample_rate=16000)

            resampled_cropped_audio = torchaudio.transforms.Resample(orig_freq=16000,
                                                                     new_freq=24000)(cropped_audio_16k.unsqueeze(0)).squeeze(0)

            sf.write(output_path, resampled_cropped_audio.numpy(), 24000)
            os.remove(temp_wav_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")