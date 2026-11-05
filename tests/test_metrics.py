import unittest
from scripts.evaluate import rouge_l


class MetricTests(unittest.TestCase):
    def test_character_lcs_f1(self):
        self.assertEqual(rouge_l('血压', '血压'), 1)
        self.assertEqual(rouge_l('', '血压'), 0)
        self.assertEqual(rouge_l('血 压', '血压'), 1)
        self.assertAlmostEqual(rouge_l('血压高', '血压低'), 2 / 3)
