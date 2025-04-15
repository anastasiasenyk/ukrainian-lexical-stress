import re

UKRAINIAN_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINIAN_LETTERS += UKRAINIAN_LETTERS.upper()

VOCAB = UKRAINIAN_LETTERS + " "

UKRAINIAN_RE = re.compile(f"[{UKRAINIAN_LETTERS}]")
UKRAINIAN_RE_DASH = re.compile(f"([{UKRAINIAN_LETTERS}])-([{UKRAINIAN_LETTERS}])")


def shift_stress_marks_right(text: str, stress_mark: str = "+") -> str:
    """
    Shifts all stress marks one position to the right in the input string.

    Args:
        text (str): The input text with misplaced stress marks.
        stress_mark (str, optional): The symbol used for stress. Default is '+'.

    Returns:
        str: Text with stress marks shifted to the right positions.
    """
    text_list = list(text)
    i = 0
    while i <= len(text_list) - 2:
        if text_list[i] == stress_mark:
            text_list[i], text_list[i + 1] = text_list[i + 1], text_list[i]
            i += 1
        i += 1
    return "".join(text_list)


def clean_text(text: str) -> str:
    """
    Cleans and normalizes Ukrainian text for processing.

    Args:
        text (str): The raw input text.

    Returns:
        str: Lowercased and cleaned text containing only Ukrainian letters and spaces.
    """
    text = text.lower()
    text = UKRAINIAN_RE_DASH.sub(r"\1 \2", text)
    text = "".join(char for char in text if char in VOCAB)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_map(text: str) -> tuple[str, list[int]]:
    """
    Cleans text and builds a map of indices to original positions.

    Args:
        text (str): The raw input text.

    Returns:
        tuple[str, list[int]]: A tuple of cleaned text and a mapping of cleaned indices to original indices.
    """
    cleaned = []
    index_map = []
    for i, char in enumerate(text):
        if UKRAINIAN_RE.match(char):
            cleaned.append(char.lower())
            index_map.append(i)
    return "".join(cleaned), index_map


def merge_texts(original: str, with_stress: str) -> str:
    """
    Merges stress annotations from a processed string back into the original.

    Args:
        original (str): Original input text.
        with_stress (str): Cleaned text with stress marks.

    Returns:
        str: Original text with stress marks applied.
    """
    norm_original, index_map = clean_and_map(original)
    norm_with_stress = "".join(c for c in with_stress.lower() if c in UKRAINIAN_LETTERS or c == "+")

    stressed_chars = {}
    i = j = 0
    while j < len(norm_with_stress):
        if j < len(norm_with_stress) and norm_with_stress[j] == "+":
            if i > 0:
                stressed_chars[i - 1] += "+"
            j += 1
        elif i < len(norm_original) and norm_with_stress[j] == norm_original[i]:
            stressed_chars[i] = norm_with_stress[j]
            i += 1
            j += 1
        else:
            j += 1

    result = list(original)
    for norm_i, char_index in enumerate(index_map):
        char = result[char_index]
        if norm_i in stressed_chars and "+" in stressed_chars[norm_i]:
            result[char_index] = char + "+"
    return "".join(result)


def split_text_by_whitespace(text: str, max_length: int = 256) -> list[str]:
    """
    Splits long text into chunks with respect to whitespace and a max length.

    Args:
        text (str): The input text.
        max_length (int, optional): Maximum length of each chunk. Default is 256.

    Returns:
        list[str]: List of text chunks.
    """
    words = text.split(" ")
    chunks = []
    current_chunk = None
    for word in words:
        if (
            len("" if current_chunk is None else current_chunk) + len(word) + (1 if current_chunk is not None else 0)
            <= max_length
        ):
            if current_chunk is None:
                current_chunk = word
            else:
                current_chunk += " " + word
        else:
            if current_chunk is not None:
                if chunks:
                    chunks.append(" " + current_chunk)
                else:
                    chunks.append(current_chunk)
            current_chunk = word
    if current_chunk is not None:
        if chunks:
            chunks.append(" " + current_chunk)
        else:
            chunks.append(current_chunk)
    return chunks
