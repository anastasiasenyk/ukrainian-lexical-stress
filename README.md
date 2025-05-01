# Implementing Ukrainian TTS with Voice Cloning


We recommend using **Python 3.10** for compatibility.
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```


# Ukrainian TTS with Voice Cloning


# Ukrainian Lexical Stress Prediction Model

We provide a model for predicting lexical stress in Ukrainian words.

### Model Overview

- **Architecture**: ByT5-based Grapheme-to-Phoneme model.
- **Training Data**: Voice of America corpus annotated with stress marks using an ASR Wav2Vec2 model. [Navigate to wav2vec2 description](#asr-wav2vec2-with-lexical-stress).


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

#### 1. Prepare the Dataset

Normalize and prepare the dataset (already annotated with stress):

```bash
python accentor_model/data/create_dataset.py
```

Convert the dataset to NeMo-compatible format:

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

> 📄 *Training script adapted from:*  
> [NVIDIA NeMo G2P Example](https://github.com/NVIDIA/NeMo/blob/main/examples/tts/g2p/g2p_train_and_evaluate.py)

---

# ASR Wav2Vec2 with Lexical Stress

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


