# Implementing Ukrainian TTS with Voice Cloning


We recommend using **Python 3.10** for compatibility.
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```


# Ukrainian TTS with Voice Cloning


# Ukrainian Lexical Stress Prediction Model

- Model architecture: Byt5 G2P

- Dataset for training: Voice of America annotated with ASR Wav2Vec2 model aimed on transcribing text with stress marks
`accentor_model/data/voice_of_america/voa_stressed_cleaned_data.csv`


### Quickstart: Run the Lexical Stress Prediction Model
```python
from accentor_model.predict_word_stress import Stressifier
from huggingface_hub import hf_hub_download

path_to_nemo_model = hf_hub_download(
    repo_id="mouseyy/stressifier-byt5-g2p-model",
    filename="T5G2P.nemo",
)

stressifier = Stressifier(path_to_nemo_model)
stressed_sentence = stressifier.stressify("Привіт, як у тебе справи?")
```

## ASR Wav2Vec

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
