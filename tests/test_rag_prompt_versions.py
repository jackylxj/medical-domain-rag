import unittest
from scripts.generate_rag_demo import prepare_prompt


class PromptVersionTests(unittest.TestCase):
    def setUp(self):
        self.row = {'question':'资料能说明什么？', 'review':'REVIEW_CANARY', 'hits':[
            {'source_id':'doc/section/1','title':'示例','section':'段落','text':'示例事实。',
             'source':'https://example.org','version':'2026-11-05'}]}

    def test_v3_preserves_evidence_and_excludes_review(self):
        v1,e1 = prepare_prompt(self.row,True,'v1')
        v3,e3 = prepare_prompt(self.row,True,'v3')
        self.assertTrue(v3.startswith(v1))
        self.assertEqual(e1,e3)
        self.assertNotIn('REVIEW_CANARY',v3)
        self.assertIn('换算',v3)
        self.assertIn('部分',v3)

    def test_v2_existing_instructions_remain(self):
        prompt,_ = prepare_prompt(self.row,True,'v2')
        self.assertIn('用1—3句话直接回答。',prompt)
        self.assertNotIn('换算',prompt)


if __name__=='__main__':unittest.main()
