# Implementing Ukrainian TTS with Voice Cloning


We recommend using **Python 3.10** for compatibility.
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

**List of content:**
- [Ukrainian TTS with Voice Cloning](#ukrainian-tts-with-voice-cloning)
- [Ukrainian Lexical Stress Prediction Model](#ukrainian-lexical-stress-prediction-model)
- [Wav2Vec2 with Lexical Stress](#wav2vec2-with-lexical-stress)
- [Ukrainian Lexical Stress Benchmark](#ukrainian-lexical-stress-benchmark)

# Ukrainian TTS with Voice Cloning

Pretrained and fine-tuned checkpoints for Ukrainian text-to-speech with voice cloning:

- **Fine-tuned model:**  
  [Ukrainian_XTTSv2_common_voice](https://huggingface.co/mouseyy/Ukrainain_XTTSv2_common_voice)

- **Baseline model:**  
  [Ukrainian_XTTSv2_common_voice_baseline](https://huggingface.co/mouseyy/Ukrainain_XTTSv2_common_voice_baseline)

You can find an example of how to run inference in the notebook:  
`tts_model/tts_model_inference.ipynb`


### 1. Prepare Dataset

- **A. Use Preprocessed Dataset**

   A ready-to-use version of the dataset is available and can be downloaded using the Hugging Face Hub:
   ```python
   from huggingface_hub import hf_hub_download
   import zipfile
   
   zip_path = hf_hub_download(
       repo_id="mouseyy/common_voice_19_uk_cropped",
       filename="wavs.zip",
       repo_type="dataset",
       local_dir="common_voice_data",
       local_dir_use_symlinks=False
   )
   
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall("common_voice_data")
   
   for split in ["train", "test", "eval"]:
       hf_hub_download(repo_id="mouseyy/common_voice_19_uk_cropped",
                       filename=f"metadata_{split}.csv",
                       repo_type="dataset",
                       local_dir="common_voice_data",
                       local_dir_use_symlinks=False)
   ```

- **B. Build the Dataset from Scratch**

   If you'd like to build the dataset manually, refer to the notebook at:
   
   ```
   tts_model/prepare_data.ipynb
   ```
   
   This notebook starts from the data in `tts_model/data/prepared`, which contains the original train/test/dev splits from Common Voice, additionally annotated using Whisper transcriptions. Word Error Rate (WER) is computed between original and Whisper outputs.
   
   To crop and resample Common Voice audio files, use the script:
   
   ```
   tts_model/crop_and_resample_cv.py
   ```

### 2. Model Training

1. Start by setting up the training environment:
   ```bash
   cd tts_model
   git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
   cd XTTSv2-Finetuning-for-New-Languages
   ```

2. Install the dependencies (recommended versions):
   - `torch==2.2.0`  
   - `torchaudio==2.2.0`
   ```bash
   pip install -r requirements.txt
   ```

3. Download the base checkpoint:
   ```bash
   python download_checkpoint.py --output_path checkpoints/
   ```

4. Extend Vocabulary for Ukrainian

   ```bash
   python extend_vocab_config.py \
       --output_path=checkpoints/ \
       --metadata_path=common_voice_data/metadata_train.csv \ # correct path to your data
       --language=uk \
       --extended_vocab_size=1000
   ```

5. Learning Rate Scheduler Modification

   We modified the training script to use `ExponentialLR` with `gamma=0.9`.  
   The updated version of the training script is located at:  
   `tts_model/xtts_modified_files/train_gpt_xtts.py`

6. Launch Training

   ```bash
   python train_gpt_xtts.py \
   --output_path checkpoints/ \
   --metadatas common_voice_data/metadata_train.csv,common_voice_data/metadata_eval.csv,uk \ # correct path to your data
   --num_epochs 100 \
   --batch_size 24 \
   --grad_acumm 2 \
   --max_text_length 200 \
   --max_audio_length 330750 \
   --weight_decay 1e-2 \
   --lr 5e-6 \
   --save_step 2000
   ```


# Ukrainian Lexical Stress Prediction Model

We provide a model for predicting lexical stress in Ukrainian words.

### Model Overview

- **Architecture**: ByT5-based Grapheme-to-Phoneme model.
- **Training Data**: Voice of America corpus annotated with stress marks using an ASR Wav2Vec2 model. [Navigate to wav2vec2 description](#wav2vec2-with-lexical-stress).

### Quickstart: Run the Lexical Stress Prediction Model
```python
from accentor_model.predict_word_stress import Stressifier
from huggingface_hub import hf_hub_download

# Download the trained model from Hugging Face Hub
path_to_nemo_model = hf_hub_download(
    repo_id="mouseyy/stressifier-byt5-g2p-model",
    filename="T5G2P.nemo",
)

# Initialize and use the stressifier
stressifier = Stressifier(path_to_nemo_model)
stressed_sentence = stressifier.stressify("Привіт, як у тебе справи?")
```

### Training Instructions

We provide a preprocessed dataset (`accentor_model/data/voice_of_america/voa_stressed_cleaned_data.csv`), already transcribed with a Wav2Vec2 model and cleaned. 
If you want to start from scratch with the raw dataset, begin at **Step 0**. Otherwise, proceed directly to **Step 1**.

### Step 0: (Optional) Prepare the Raw Dataset

1. **Download the dataset** 

   Download from [Zenodo](https://zenodo.org/records/7405411) and save to the `accentor_model/data/prepare_dataset/` directory and navigate to it.
2. **Convert "voa_clean.jsonl" to "voa_clean.csv"**

    ```bash
    python convert_to_csv.py
    ```
3. **Re-transcribe with Whisper to clean audio** 

    ```bash
    python transcribe_with_whisper.py
    ```
4. **Clean the dataset using original and Whisper transcriptions**  

    ``` bash
    python clean_dataset.py
    ```
5. **Transcribe cleaned dataset with Wav2Vec2 to form a synthetic dataset with stress marks**  

    ``` bash
    python transcribe_with_whisper.py
    ```
6. **Clean dataset after Wav2Vec transcriptions (data provided in 'accentor_model/data/raw')**
    ``` bash
    python cleaned_stressed_dataset.py
    ```


#### 1. Prepare the Dataset

From the project root directory:

1. Normalize and prepare the dataset (already annotated with stress):

    ```bash
    python accentor_model/data/create_dataset.py
    ```

2. Convert the dataset to NeMo-compatible format:

    ```bash
    python accentor_model/byt5_g2p/dataset_to_nemo_format.py
    ```

#### 2. Train the Model

Navigate to the training directory and run the training script:

```bash
cd accentor_model/byt5_g2p/

python g2p_train_and_evaluate.py \
    --config-path=$PWD \
    --config-name=g2p_t5 \
    model.train_ds.manifest_filepath=train.json \
    model.validation_ds.manifest_filepath=eval.json \
    model.test_ds.manifest_filepath=eval.json \
    trainer.devices=1 \
    do_training=True \
    do_testing=True
```

> *Training script adapted from:*  
> [NVIDIA NeMo G2P Example](https://github.com/NVIDIA/NeMo/blob/main/examples/tts/g2p/g2p_train_and_evaluate.py)

---

# Wav2Vec2 with Lexical Stress

The model transcribes Ukrainian speech with incorporated lexical stress directly in the text.  
Fine-tuned model is available on Hugging Face: [mouseyy/uk_wav2vec2_with_stress_mark](https://huggingface.co/mouseyy/uk_wav2vec2_with_stress_mark)

### Training Data

The model was trained using the [Common Voice](https://commonvoice.mozilla.org/) corpus, annotated with stress information from:
- [Ukrainian Word Stress](https://github.com/lang-uk/ukrainian-word-stress)  
- [Ukrainian Accentor](https://github.com/egorsmkv/ukrainian-accentor)

### 1. Prepare the Dataset
   ```bash
   python wav2vec/data/setup_dataset.py
   ```

### 2. Train the Model
   ```bash
   bash wav2vec/train/run_tuning_script.sh
   ```

> *Training script adapted from:*  
> [Transformers Pytorch Examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py)

---

# Ukrainian Lexical Stress Benchmark

The **Ukrainian Lexical Stress Benchmark** is a manually annotated dataset designed to evaluate lexical stress prediction systems with a sentence-level context.

**Dataset Location:**  
```
lexical_stress_benchmark/data/lexical_stress_dataset.csv
```

Each sentence contains a stress mark (`+`) following the stressed vowel. The dataset includes columns:

- `StressedSentence`: The sentence with stress annotations.
- `Source`: Origin of the sentence (either `wiki`, `plug`, or `custom`).

### Example
```csv
У+ ва+зі стоя+ли кві+ти.,custom
```

### Dataset Composition

| Statistic                                                                 | Count |
|--------------------------------------------------------------------------|-------:|
| Total number of sentences                                                | 1,026 |
| Unique word forms (including grammatical inflections, derivations, etc.) | 6,439 |
| Unique words with stress ambiguity (due to meaning or inflections)       |   640 |
| Unique words with at least two stress forms in the dataset               |   296 |


- **Wikipedia** (300 sentences) - formal, encyclopedic style \[[wikimedia](https://dumps.wikimedia.org)\]
- **Pluperfect GRAC** (438 sentences) - fiction, journalism, poetry, etc. \[[pluperfect_grac](https://github.com/Dandelliony/pluperfect_grac)\]
- **Custom** (288 sentences) - manually written to balance ambiguous stress patterns.


### Evaluation Metrics

The benchmark evaluates stress prediction system systems using:

- **Word-Level Accuracy**
- **Sentence-Level Accuracy**
- **Unambiguous Word Accuracy**
- **Ambiguous Word Accuracy**
- **Macro-Average F1 (Ambiguous Word Pairs)**


### Quickstart: Run the Benchmark

```python
from lexical_stress_benchmark.benchmark import evaluate_stressification

def custom_stressify(text):
    """
    Annotates the input text with stress marks.

    The function should return the input `text` with a '+' symbol placed immediately 
    after the stressed vowel in each stressed word.

    Args:
        text (str): A sentence in Ukrainian.

    Returns:
        str: The sentence with stress marks added.
    """
    # your implementation
    return text 

accuracies = evaluate_stressification(custom_stressify)
for metric, value in accuracies.items():
    print(f"{metric:40} {value * 100:.2f}%")
```


# References

### Dataset Sources

- **Common Voice: A Massively-Multilingual Speech Corpus**. Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M. Tyers, Gregor Weber. In *Proceedings of the 12th Conference on Language Resources and Evaluation (LREC)*, 2020. [https://aclanthology.org/2020.lrec-1.520/](https://aclanthology.org/2020.lrec-1.520/)

- **Voice of America: Ukrainian ASR Dataset of Broadcast Speech**. Yehor Smoliakov, 2022. Zenodo. DOI: [https://doi.org/10.5281/zenodo.7405411](https://doi.org/10.5281/zenodo.7405411)

- **PluG: Corpus of Old Ukrainian Texts**. Maria Shvedova and Arsenii Lukashevskyi, 2024. Electronic resource: Kharkiv, Jena. Available at: [https://github.com/Dandelliony/pluperfect_grac](https://github.com/Dandelliony/pluperfect_grac)

- **Wikimedia Downloads**. Provided by the Wikimedia Foundation. Available at: [https://dumps.wikimedia.org](https://dumps.wikimedia.org) (Accessed: 2024-12-21).

- **Dictionaries of Ukraine Online**. Ukrainian Lingua-Information Foundation, NAS of Ukraine, 2008. Available at: [https://lcorp.ulif.org.ua/dictua/](https://lcorp.ulif.org.ua/dictua/)


### Models

- **XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model**. Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, Julian Weber. In *Interspeech*, 2024. [arXiv:2406.04904](https://arxiv.org/abs/2406.04904)

- **ByT5 for Massively Multilingual Grapheme-to-Phoneme Conversion**. Jian Zhu, Cong Zhang, David Jurgens. In *Interspeech*, 2022. [arXiv:2204.03067](https://arxiv.org/abs/2204.03067)

- **Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli. In *Conference on Neural Information Processing Systems*, 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)


