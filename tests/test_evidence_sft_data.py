import unittest
import tempfile,subprocess,sys,json,hashlib
from pathlib import Path
from scripts.prepare_evidence_sft import validate_rows

def row(id='a',split='train',url='https://example.org/a'):
 return dict(id=id,split=split,question='资料允许什么？',answer='可以选择甲。[1]',kind='answerable',evidence=[dict(citation=1,source=url,source_id=url+'/1',text='可以选择甲。')],supports=[dict(citation=1,quote='可以选择甲。')],missing='')
class DatasetChecks(unittest.TestCase):
 def test_export_is_lf_and_manifest_matches(self):
  root=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as tmp:
   output=Path(tmp)/'export'
   subprocess.run([sys.executable,'-m','scripts.prepare_evidence_sft','--output',str(output)],cwd=root,check=True,capture_output=True)
   manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
   for name,meta in manifest['files'].items():
    raw=(output/name).read_bytes()
    self.assertNotIn(b'\r',raw)
    self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['sha256'])
    self.assertEqual(len(raw.splitlines()),meta['count'])
 def test_valid(self):validate_rows([row()])
 def test_source_leak(self):
  b=row('b','validation');b['question']='另一个问题？'
  with self.assertRaises(ValueError):validate_rows([row(),b])
 def test_citation(self):
  a=row();a['answer']='可以选择甲。[2]'
  with self.assertRaises(ValueError):validate_rows([a])
 def test_fabricated_support(self):
  a=row();a['supports'][0]['quote']='可以选择乙。'
  with self.assertRaises(ValueError):validate_rows([a])
 def test_refusal_has_citation(self):
  a=row();a.update(kind='unanswerable',missing='信息',answer='资料不足。[1]',supports=[])
  with self.assertRaises(ValueError):validate_rows([a])
 def test_duplicate_question(self):
  with self.assertRaises(ValueError):validate_rows([row(),row('b')])
 def test_text_leak_new_url(self):
  b=row('b','validation','https://example.org/b');b['question']='另一问题？'
  with self.assertRaises(ValueError):validate_rows([row(),b])
if __name__=='__main__':unittest.main()
