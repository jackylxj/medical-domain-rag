import tempfile
import unittest
from pathlib import Path
from scripts.build_text_index import split_text, read_document


class CorpusTests(unittest.TestCase):
    def test_offsets_cover_text_and_overlap(self):
        text = '甲乙丙丁。' * 100
        spans = split_text(text, 100, 20)
        covered = set()
        for start, end in spans:
            self.assertLessEqual(end-start, 100)
            covered.update(range(start,end))
        self.assertEqual(covered, set(range(len(text))))
        for a,b in zip(spans,spans[1:]):
            self.assertEqual(a[1]-b[0],20)

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            split_text('abc', 10, 10)

    def test_metadata_required_and_duplicate_heading_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.txt'
            p.write_text('无来源文本',encoding='utf-8')
            with self.assertRaises(ValueError):read_document(p)
            p.write_text('# 标题\n来源：https://example.org/source\n版本：2026-11-05\n## 主题\n内容\n## 主题\n重复标题',encoding='utf-8')
            with self.assertRaises(ValueError):read_document(p)


if __name__=='__main__':unittest.main()
