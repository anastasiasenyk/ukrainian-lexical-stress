import re

from accentor_model.predict_word_stress.model_utils import load_g2p_model, text_to_phonemes
from accentor_model.predict_word_stress.text_utils import (UKRAINIAN_LETTERS, clean_text, merge_texts,
                                                           shift_stress_marks_right, split_text_by_whitespace)


class Stressifier:
    def __init__(self, model_path: str):
        self.model = load_g2p_model(model_path)

    def stressify(self, text: str) -> str:
        """
        Applies stress marks to Ukrainian text using a G2P model.

        Args:
            text (str): The original Ukrainian text.

        Returns:
            str: Text with stress marks inserted after the stressed vowels.
        """
        chunks = split_text_by_whitespace(text)
        stressified_chunks = []

        for original_chunk in chunks:
            cleaned = clean_text(original_chunk)
            match = re.search(rf"[{UKRAINIAN_LETTERS}]", cleaned)

            if not match:
                stressified_chunks.append(original_chunk)
                continue

            prefix = cleaned[: match.start()]
            chunk = cleaned[match.start() :]
            ends_with_dot = chunk.endswith(".")

            if not ends_with_dot:
                chunk += "."

            stressed = text_to_phonemes(chunk, self.model)
            stressed = shift_stress_marks_right(stressed)

            if stressed.endswith(".") and not ends_with_dot:
                stressed = stressed[:-1]

            final_word = prefix + stressed
            merged = merge_texts(original_chunk, final_word)

            stressified_chunks.append(merged)

        return "".join(stressified_chunks)
