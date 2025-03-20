import unittest
from benchmark import evaluate_stress_sentence_level

from benchmark.benchmark.sentence_stressification import SentenceMismatchError, WordDetectionError
from benchmark.benchmark.accuracy import SentenceAccuracy


class TestEvaluateStressWordLevel(unittest.TestCase):
    def test_case_1(self):
        obj1 = SentenceAccuracy()
        obj1.word_count = 2
        obj1.correct_word_count = 2
        obj1.heteronym_count = 1
        obj1.correct_heteronym_count = 1

        obj2 = evaluate_stress_sentence_level("За+вжди+ ра+ді!", "За+вжди ра+ді!")
        self.assertEqual(obj1, obj2)

    def test_case_2(self):
        obj1 = SentenceAccuracy()
        obj1.word_count = 1
        obj1.correct_word_count = 1
        obj1.heteronym_count = 0
        obj1.correct_heteronym_count = 0

        obj2 = evaluate_stress_sentence_level("Привіт, сві+те!", "Приві+т, сві+те!")
        self.assertEqual(obj1, obj2)

    def test_case_3(self):
        obj1 = SentenceAccuracy()
        obj1.word_count = 2
        obj1.correct_word_count = 0
        obj1.heteronym_count = 0
        obj1.correct_heteronym_count = 0

        obj2 = evaluate_stress_sentence_level("Приві+т, сві+те!", "При+віт, світе+!")
        self.assertEqual(obj1, obj2)

    def test_case_error_1(self):
        with self.assertRaises(SentenceMismatchError) as context:
            evaluate_stress_sentence_level("Приві+т, сві+те!", "Приві+т, мій світе+!")

        self.assertEqual(str(context.exception), "The number of words in the sentences does not match.")

    def test_case_error_2(self):
        with self.assertRaises(WordDetectionError) as context:
            evaluate_stress_sentence_level("Привіт, світе!", "Приві+т, світе+!")

        self.assertEqual(str(context.exception), "No words were detected in the sentence.")
