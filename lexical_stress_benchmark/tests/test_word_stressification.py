import unittest

from lexical_stress_benchmark.benchmark.sentence_stressification import evaluate_stress_word_level


class TestEvaluateStressWordLevel(unittest.TestCase):
    def test_case_1(self):
        # Correctly stressed word matches
        self.assertTrue(evaluate_stress_word_level("За+вжди+", "За+вжди"))
        self.assertTrue(evaluate_stress_word_level("За+вжди+", "Завжди+"))
        self.assertTrue(evaluate_stress_word_level("За+вжди+", "За+вжди+"))

    def test_case_2(self):
        # No stress in the evaluated word, stress doesn't match
        self.assertFalse(evaluate_stress_word_level("За+вжди+", "Завжди"))
        self.assertFalse(evaluate_stress_word_level("За+вжди+", "За+вж+ди"))

    def test_case_3(self):
        # One-vowel word
        self.assertTrue(evaluate_stress_word_level("так", "та+к"))
        self.assertTrue(evaluate_stress_word_level("так", "так"))

    def test_case_4(self):
        # Stress is incorrect in evaluated word
        self.assertFalse(evaluate_stress_word_level("Кві+тка", "Квітка+"))

    def test_case_5(self):
        # The word letter were changed
        self.assertFalse(evaluate_stress_word_level("світе+", "світу+"))

    def test_case_6(self):
        # The word letter were changed
        self.assertTrue(evaluate_stress_word_level("сві+то+ви+й", "сві+то+вий"))
        self.assertFalse(evaluate_stress_word_level("сві+тови+й", "сві+то+вий"))
        self.assertFalse(evaluate_stress_word_level("сві+товий", "сві+то+вий"))
