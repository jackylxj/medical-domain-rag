import json
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_data import prepare
from scripts.validate_data import validate


class DataTests(unittest.TestCase):
    def test_reproducible_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'raw.jsonl'
            source.write_text(''.join(json.dumps({'id': i, 'question': f'q{i}', 'answer': 'a'}) + '\n' for i in range(25)), encoding='utf-8')
            self.assertEqual(validate(source)['records'], 25)
            prepare(source, root / 'a', 10, 42)
            prepare(source, root / 'b', 10, 42)
            self.assertEqual((root / 'a/train.jsonl').read_bytes(), (root / 'b/train.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(source, root / 'a', 10, 42)

    def test_broken_data_never_creates_training_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'bad.jsonl'
            for content in [b'{"question":"q","answer":"a"}\n\xe4\xb8', b'{"question":"q","answer":""}\n']:
                source.write_bytes(content)
                with self.assertRaises((ValueError, UnicodeError)):
                    prepare(source, root / 'out', 1, 42)
                self.assertFalse((root / 'out/train.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
