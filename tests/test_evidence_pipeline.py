import unittest
from scripts.evidence_pipeline_experiment import validate_quotes, validate_claims, accept_review

class EvidenceChecks(unittest.TestCase):
    evidence=[{'citation':1,'text':'没有添加糖或过量钠的冷冻蔬果也可以选择。'}]
    def test_fabricated_quote_rejected(self):
        with self.assertRaises(ValueError):
            validate_quotes({'quotes':[{'citation':1,'quote':'所有冷冻食品都健康'}]},self.evidence)
    def test_exact_quote_allowed(self):
        self.assertEqual(len(validate_quotes({'quotes':[{'citation':1,'quote':'冷冻蔬果也可以选择'}]},self.evidence)),1)
    def test_unknown_citation_rejected(self):
        with self.assertRaises(ValueError):
            validate_claims({'claims':[{'text':'可以选择','citations':[2]}],'missing':''},[{'citation':1,'quote':'可以选择'}])
    def test_missing_review_rejected(self):
        with self.assertRaises(ValueError):accept_review({'claims':[]},[{'text':'可以选择','citations':[1]}])
    def test_negative_or_string_verdict_not_accepted(self):
        for verdict in (False,'true'):
            self.assertFalse(accept_review({'claims':[{'index':0,'supported':verdict,'reason':'test'}]},[{'text':'可以选择','citations':[1]}]))
    def test_true_complete_review_accepted(self):
        self.assertTrue(accept_review({'claims':[{'index':0,'supported':True,'reason':'直接依据'}]},[{'text':'可以选择','citations':[1]}]))
    def test_empty_claim_not_allowed(self):
        with self.assertRaises(ValueError):validate_claims({'claims':[{'text':'','citations':[1]}],'missing':''},[{'citation':1,'quote':'可以选择'}])

if __name__=='__main__':unittest.main()
