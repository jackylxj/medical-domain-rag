import unittest
from scripts.evaluate_candidate_baseline import make_payload, select_development, citation_stats

class CandidateBaselineTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='d1',split='validation',question='问题',answer='秘密答案',kind='answerable',
            evidence=[dict(citation=1,title='标题',section='段落',text='公开证据')])
    def test_reference_never_in_prompt(self):
        p=make_payload(self.row,'base')
        self.assertNotIn('秘密答案',str(p))
        self.assertIn('公开证据',p['messages'][1]['content'])
    def test_only_model_differs(self):
        a,b=make_payload(self.row,'base'),make_payload(self.row,'lora')
        a.pop('model');b.pop('model');self.assertEqual(a,b)
    def test_no_train_or_reserved_questions(self):
        rows=[self.row,dict(self.row,id='t',split='train'),dict(self.row,id='x',split='test_reserved')]
        self.assertEqual([r['id'] for r in select_development(rows)],['d1'])
    def test_citation_existence_is_not_correctness(self):
        self.assertEqual(citation_stats('回答[2]',self.row['evidence']),{'numbers':[2],'invalid':[2]})
    def test_strict_changes_only_user_instruction(self):
        short=make_payload(self.row,'base')
        strict=make_payload(self.row,'base','strict')
        self.assertTrue(strict['messages'][1]['content'].startswith(short['messages'][1]['content']))
        self.assertNotIn('秘密答案',str(strict))
        self.assertEqual(short['messages'][0],strict['messages'][0])
        short.pop('messages');strict.pop('messages');self.assertEqual(short,strict)
    def test_strict_is_identical_for_models(self):
        a,b=make_payload(self.row,'base','strict'),make_payload(self.row,'lora','strict')
        a.pop('model');b.pop('model');self.assertEqual(a,b)
    def test_rejects_unknown_prompt(self):
        with self.assertRaises(ValueError):make_payload(self.row,'base','typo')

if __name__=='__main__':unittest.main()
