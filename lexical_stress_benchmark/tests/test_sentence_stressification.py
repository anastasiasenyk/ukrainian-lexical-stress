import unittest

from lexical_stress_benchmark import evaluate_stress_sentence_level
from lexical_stress_benchmark.benchmark import SentenceAccuracy
from lexical_stress_benchmark.benchmark.sentence_stressification import SentenceMismatchError, WordDetectionError


class TestEvaluateStressWordLevel(unittest.TestCase):
    def test_case_1(self):
        expected = SentenceAccuracy()
        expected.total_words = 2
        expected.correctly_stressified_words = 2
        expected.total_heteronyms = 1
        expected.correctly_stressified_heteronyms = 1
        expected.total_unambiguous_words = 0
        expected.correctly_stressified_unambiguous = 0

        result = evaluate_stress_sentence_level("За+мо+к закри+тий!", "За+мок закри+тий!")
        self.assertEqual(expected, result)

    def test_case_2(self):
        expected = SentenceAccuracy()
        expected.total_words = 1
        expected.correctly_stressified_words = 1
        expected.total_heteronyms = 0
        expected.correctly_stressified_heteronyms = 0
        expected.total_unambiguous_words = 0
        expected.correctly_stressified_unambiguous = 0

        result = evaluate_stress_sentence_level("Привіт, сві+те!", "Приві+т, сві+те!")
        self.assertEqual(expected, result)

    def test_case_3(self):
        expected = SentenceAccuracy()
        expected.total_words = 2
        expected.correctly_stressified_words = 0
        expected.total_heteronyms = 0
        expected.correctly_stressified_heteronyms = 0
        expected.total_unambiguous_words = 0
        expected.correctly_stressified_unambiguous = 0

        result = evaluate_stress_sentence_level("Приві+т, сві+те!", "При+віт, світе+!")
        self.assertEqual(expected, result)

    def test_case_error_1(self):
        with self.assertRaises(SentenceMismatchError) as context:
            evaluate_stress_sentence_level("Приві+т, сві+те!", "Приві+т, мій світе+!")

        self.assertEqual(str(context.exception), "The number of words in the sentences does not match.")

    def test_case_error_2(self):
        with self.assertRaises(WordDetectionError) as context:
            evaluate_stress_sentence_level("Привіт, світе!", "Приві+т, світе+!")

        self.assertEqual(str(context.exception), "No words were detected in the sentence.")
