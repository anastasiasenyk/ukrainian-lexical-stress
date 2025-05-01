import re
from ukrainian_word_stress import Stressifier, OnAmbiguity
import ukrainian_accentor as accentor
from datasets import load_dataset
import json

common_voice_train = load_dataset("mozilla-foundation/common_voice_17_0", "uk", split="train+validation", trust_remote_code=True)
common_voice_test = load_dataset("mozilla-foundation/common_voice_17_0", "uk", split="test", trust_remote_code=True)
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# Constants
STRESS_MARK = "+"
UKRAINIAN_VOWELS = "аеєиіїоуюя"

# Initialize stressifier
stressifier = Stressifier(stress_symbol=STRESS_MARK, on_ambiguity=OnAmbiguity.Skip)

# Common text normalization
REPLACEMENTS = {
    "՚": "’",
    "`": "’", 
    '"': "’",
    "'": "’",
    'a': 'а',  # English 'a' -> Ukrainian 'а'
    'e': 'е',  # English 'e' -> Ukrainian 'е'
    'i': 'і',  # English 'i' -> Ukrainian 'і'
    'o': 'о',  # English 'o' -> Ukrainian 'о'
    'p': 'р',  # English 'p' -> Ukrainian 'р'
    'k': 'к',  # English 'k' -> Ukrainian 'к'
    'ы': 'и',
    'y': 'у'
    
}

MANUAL_CORRECTIONS = {
    "його": "його" + STRESS_MARK,
    "до мене": "до ме" + STRESS_MARK + "не",
    "до себе": "до се" + STRESS_MARK + "бе",
    "до тебе": "до те" + STRESS_MARK + "бе",
}


def normalize_text(word):
    for old, new in REPLACEMENTS.items():
        word = word.replace(old, new)
    return word


def clean_text(sentence):
    words = re.split(r'[ ,.?!;:«»()[\]-]+', sentence.lower())
    pattern = r'^[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюя]+|[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюя]+$'
    return " ".join(normalize_text(re.sub(pattern, '', word)) for word in words if word)


def remove_special_characters(batch):
    batch["sentence"] = clean_text(batch["sentence"])
    return batch


def apply_manual_corrections(sentence):
    for original, stressed in MANUAL_CORRECTIONS.items():
        sentence = sentence.replace(original, stressed)
    return sentence


def shift_stress_marks(text):
    text_list = list(text)
    for i in range(1, len(text_list)):
        if text_list[i] == STRESS_MARK and text_list[i - 1] in UKRAINIAN_VOWELS:
            text_list[i], text_list[i - 1] = text_list[i - 1], text_list[i]  # Swap
    return ''.join(text_list)


def add_stress_to_single_vowel_word(word):
    for i, char in enumerate(word):
        if char in UKRAINIAN_VOWELS:
            return word[:i] + STRESS_MARK + word[i:]
    return word


def apply_ukrainian_accentor(sentence):
    words = sentence.split()
    updated_words = []
    for word in words:
        if STRESS_MARK in word:
            updated_words.append(word)
        else:
            word = add_stress_to_single_vowel_word(word) if sum(char in UKRAINIAN_VOWELS for char in word) == 1 \
                else accentor.process(word, mode='plus')
            updated_words.append(word)
    return " ".join(updated_words)


def retain_first_stress(word):
    found_stress = False
    result = []
    for char in word:
        if char == STRESS_MARK:
            if not found_stress:
                result.append(char)
                found_stress = True  # Mark that we've found the first stress
        else:
            result.append(char)  # Add non-stressed characters

    return ''.join(result)


def apply_ukrainian_word_stress(sentence):
    sentence = stressifier(sentence)
    words = sentence.split()
    words = [retain_first_stress(word) for word in words]
    sentence = " ".join(words)
    return sentence


def add_stress(sentence):
    sentence = sentence.replace('́', STRESS_MARK)  # Normalize stress mark
    sentence = apply_manual_corrections(sentence)

    sentence = sentence.replace(STRESS_MARK, '́')  # for lang_uk
    sentence = apply_ukrainian_word_stress(sentence)
    sentence = sentence.replace('́', STRESS_MARK)

    sentence = shift_stress_marks(sentence)
    sentence = apply_ukrainian_accentor(sentence)
    return sentence


def add_stress_batch(batch):
    batch["sentence"] = add_stress(batch["sentence"])
    return batch


# Function to update the sentence
def update_sentence(example):
    key = example['path'].split('.')[-2].split('/')[-1]
    example['sentence'] = add_stress(example['sentence'].lower())  # Assuming add_stress is defined
    return example

common_voice_test_norm = common_voice_test.map(update_sentence)
common_voice_train_norm = common_voice_train.map(update_sentence)


replace_map = {
    'a': 'а',  # English 'a' -> Ukrainian 'а'
    'e': 'е',  # English 'e' -> Ukrainian 'е'
    'i': 'і',  # English 'i' -> Ukrainian 'і'
    'o': 'о',  # English 'o' -> Ukrainian 'о'
    'p': 'р',  # English 'p' -> Ukrainian 'р'
    'k': 'к',  # English 'k' -> Ukrainian 'к'
    'ы': 'и',
    'y': 'у', # English 'y' -> Ukrainian 'у'
    'c': 'с', # English 'c' -> Ukrainian 'с'
    '»': '',
    '«': ''
}

def replace_letters(text):
    for eng, ukr in replace_map.items():
        text = text.replace(eng, ukr)
    return text

def apply_replacement(dataset):
    return dataset.map(lambda example: {'sentence': replace_letters(example['sentence'])})

common_voice_train_norm = apply_replacement(common_voice_train_norm)
common_voice_test_norm = apply_replacement(common_voice_test_norm)


common_voice_test_norm.save_to_disk('./wav2vec/data/common_voice_test_data')
common_voice_train_norm.save_to_disk('./wav2vec/data/common_voice_train_data')


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train_norm.map(
  extract_all_chars, batched=True,
  batch_size=-1, keep_in_memory=True,
  remove_columns=common_voice_train_norm.column_names
)
vocab_test = common_voice_test_norm.map(
  extract_all_chars, batched=True,
  batch_size=-1, keep_in_memory=True,
  remove_columns=common_voice_test_norm.column_names
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

print("Len vocab: ", len(vocab_dict))
print(vocab_dict)

with open('./wav2vec/data/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
