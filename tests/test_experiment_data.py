import unittest
from scripts.prepare_experiment import normalize, group_questions, stratified_take


class ExperimentDataTests(unittest.TestCase):
    def test_normalization_and_near_duplicates(self):
        rows = [
            {'question': '请问：持续头痛超过三天应该去医院哪个科室就诊？'},
            {'question': '请问持续头痛超过三天应该去医院哪个科室就诊'},
            {'question': '请问持续头痛超过三天应该去医院什么科室就诊'},
            {'question': '什么是血压'}]
        groups, excluded = group_questions(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[0], groups[3])
        self.assertFalse(excluded)

    def test_split_selection_is_disjoint_and_exact(self):
        import random
        rows = [{'label': str(i % 3), 'id': i} for i in range(90)]
        chosen, rest = stratified_take(rows, 20, random.Random(42))
        self.assertEqual(len(chosen), 20)
        self.assertFalse({r['id'] for r in chosen} & {r['id'] for r in rest})
        self.assertEqual(len(rest), 70)
