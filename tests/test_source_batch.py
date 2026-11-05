import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_source_batch import build_batch


class SourceBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        body = '## 定义\n\n这是原文中的完整事实。\n\n这是第二个事实。'
        (self.root / 'a.md').write_text(body, encoding='utf-8')
        self.manifest = {'sources': [{
            'id': 'a', 'source_group': 'who:a', 'split': 'train',
            'url': 'https://www.who.int/zh/news-room/fact-sheets/detail/a',
            'title': '资料A', 'clean_relative_path': 'a.md',
            'clean_sha256': hashlib.sha256((self.root / 'a.md').read_bytes()).hexdigest()
        }]}
        self.spec = {'facts': [{'id': 'a1', 'source_id': 'a', 'quote': '这是原文中的完整事实。'}],
                     'examples': [{'id': 'q1', 'kind': 'answerable', 'question': '资料说明什么？',
                                   'evidence_ids': ['a1'], 'support_ids': ['a1'],
                                   'answer': '这是原文中的完整事实。[1]', 'missing': ''}]}

    def build(self):
        return build_batch(self.spec, self.manifest, self.root)

    def test_records_have_verifiable_offsets(self):
        rows, facts = self.build()
        f = facts[0]
        body = (self.root / 'a.md').read_text(encoding='utf-8')
        self.assertEqual(body[f['start_char']:f['end_char']], f['quote'])
        self.assertEqual(rows[0]['split'], 'train')

    def test_rejects_nonverbatim_support(self):
        self.spec['facts'][0]['quote'] = '原文中不存在的支持证据'
        with self.assertRaises(ValueError): self.build()

    def test_rejects_reserved_test_source(self):
        self.manifest['sources'][0]['split'] = 'test_reserved'
        with self.assertRaises(ValueError): self.build()

    def test_rejects_tampered_snapshot(self):
        (self.root / 'a.md').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError): self.build()

    def test_rejects_cross_split_distractor(self):
        b = dict(self.manifest['sources'][0], id='b', source_group='who:b', split='development')
        self.manifest['sources'].append(b)
        self.spec['facts'].append({'id':'b1','source_id':'b','quote':'这是第二个事实。'})
        self.spec['examples'][0]['evidence_ids'].append('b1')
        with self.assertRaises(ValueError): self.build()

    def test_rejects_quote_with_lost_qualifier(self):
        self.spec['facts'][0]['quote'] = '中的完整事实'
        self.spec['facts'][0]['start_char'] = 0
        # Extracts must start/end at a sentence or paragraph boundary.
        with self.assertRaises(ValueError): self.build()


if __name__ == '__main__': unittest.main()
