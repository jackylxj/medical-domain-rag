import unittest
from scripts.collect_evidence_sources import source_group,validate_registry
class SourceRegistryChecks(unittest.TestCase):
 def test_language_version_alias(self):
  self.assertEqual(source_group('https://www.who.int/en/news-room/fact-sheets/detail/anaemia?x=1'),source_group('https://www.who.int/zh/news-room/fact-sheets/detail/anaemia/'))
 def test_conflicting_split(self):
  rows=[dict(url='https://www.who.int/zh/news-room/fact-sheets/detail/anaemia',id='a',split='train'),dict(url='https://www.who.int/en/news-room/fact-sheets/detail/anaemia',id='b',split='test_reserved')]
  with self.assertRaises(ValueError):validate_registry(dict(sources=rows,old_development_groups=[]))
 def test_old_source_rejected(self):
  with self.assertRaises(ValueError):validate_registry(dict(sources=[dict(url='https://www.who.int/zh/news-room/fact-sheets/detail/diabetes',id='a',split='train')],old_development_groups=['diabetes']))
 def test_external_host_rejected(self):
  with self.assertRaises(ValueError):source_group('https://example.org/anaemia')
if __name__=='__main__':unittest.main()
