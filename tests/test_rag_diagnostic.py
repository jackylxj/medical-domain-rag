import unittest
from scripts.diagnose_rag_refusal import build_cases, TARGET, CONTROL


class DiagnosticTests(unittest.TestCase):
    def test_one_factor_pairs_and_no_mutation(self):
        hits=[{'source_id':'diet/蔬果选择/0001'}, {'source_id':'other/2'}, {'source_id':'other/3'}]
        source=[{'question':TARGET,'hits':hits}, {'question':CONTROL,'hits':[{'source_id':str(i)} for i in range(3)]}]
        rows,cases=build_cases(source)
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0]['question'],rows[1]['question'])
        self.assertEqual(rows[1]['hits'],rows[0]['hits'][:1])
        self.assertEqual(rows[1]['hits'],rows[2]['hits'])
        self.assertEqual(rows[2]['question'],rows[3]['question'])
        self.assertEqual(rows[0]['hits'],rows[3]['hits'])
        self.assertEqual(rows[4]['question'],rows[5]['question'])
        self.assertEqual(rows[5]['hits'],rows[4]['hits'][:1])
        rows[0]['hits'][0]['source_id']='modified'
        self.assertEqual(hits[0]['source_id'],'diet/蔬果选择/0001')
        self.assertEqual(len({c['id'] for c in cases}),6)

    def test_missing_target_fails(self):
        with self.assertRaises(ValueError):build_cases([])


if __name__=='__main__':unittest.main()
