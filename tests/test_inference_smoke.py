import unittest
from pathlib import Path
from scripts.infer_smoke import build_model_args


class InferenceConfigTests(unittest.TestCase):
    def test_base_never_loads_adapter(self):
        args = build_model_args('base', '/models/qwen', Path('/outputs/adapter'))
        self.assertNotIn('adapter_name_or_path', args)
        self.assertEqual(args['template'], 'qwen')

    def test_lora_uses_explicit_adapter_and_same_base(self):
        base = build_model_args('base', '/models/qwen', Path('/outputs/adapter'))
        tuned = build_model_args('lora', '/models/qwen', Path('/outputs/adapter'))
        self.assertEqual(tuned.pop('adapter_name_or_path'), str(Path('/outputs/adapter').resolve()))
        self.assertEqual(base, tuned)
