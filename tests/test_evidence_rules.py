import unittest
from scripts.evidence_rules import inspect_claim, split_claim, ratio_proof
class DeterministicChecks(unittest.TestCase):
 def check(self,text,source,question=''):
  return inspect_claim({'text':text,'citations':[1]},[{'citation':1,'text':source}],question)
 def test_boundary_change(self):
  self.assertIn('comparison_changed',self.check('食盐应不超过5克。','食盐应低于5克。')['issues'])
 def test_boundary_preserved(self):
  self.assertNotIn('comparison_changed',self.check('食盐应低于5克。','食盐应低于5克。')['issues'])
 def test_other_number_boundary(self):
  self.assertIn('comparison_changed',self.check('数量至多12毫克。','数量少于12毫克。')['issues'])
 def test_unit_changed(self):
  self.assertIn('quantity_not_in_evidence',self.check('应摄入5毫克。','应摄入5克。')['issues'])
 def test_correct_ratio(self):
  self.assertIsNotNone(ratio_proof('800毫克钠约等于2克食盐。','1克食盐约对应400毫克钠。'))
 def test_wrong_ratio(self):
  self.assertIsNone(ratio_proof('800毫克钠约等于3克食盐。','1克食盐约对应400毫克钠。'))
 def test_unseen_ratio_decimal(self):
  self.assertIsNotNone(ratio_proof('125毫克钠约等于0.5克食盐。','2克食盐约对应500毫克钠。'))
 def test_conflicting_ratios(self):
  self.assertIsNone(ratio_proof('800毫克钠约等于2克食盐。','1克食盐约对应400毫克钠，1克食盐约对应200毫克钠。'))
 def test_negated_relation_not_proven(self):
  self.assertIsNone(ratio_proof("800毫克钠约等于2克食盐。","不能认为1克食盐约对应400毫克钠。"))
 def test_extra_advice_not_proven(self):
  self.assertIsNone(ratio_proof('800毫克钠约等于2克食盐。建议多吃。','1克食盐约对应400毫克钠。'))
 def test_split_chinese_and_decimal(self):
  self.assertEqual(split_claim('含0.5克。还可选择低钠产品！'),['含0.5克。','还可选择低钠产品！'])
 def test_population(self):
  self.assertIn('population_mismatch',self.check('成年人应低于5克。','成年人应低于5克。','儿童可以吃多少？')['issues'])
 def test_unknown_citation(self):
  self.assertIn('unknown_citation',inspect_claim({'text':'可以选择。','citations':[2]},[{'citation':1,'text':'可以选择。'}])['issues'])
 def test_exact_quote_does_not_prove_relevance(self):
  self.assertEqual(self.check('可以选择冷冻蔬果。','可以选择冷冻蔬果。')['support'],'exact_text_only')
if __name__=='__main__':unittest.main()
