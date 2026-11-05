"""Reload base or LoRA in a separate process; generation smoke test, not evaluation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = '请用通俗语言解释什么是血压，以及收缩压和舒张压分别代表什么。'
SYSTEM = '你是医学知识科普助手。请准确、简洁地回答；不确定时明确说明，不对个人作出诊断。'


def build_model_args(mode, model, adapter):
    args = {'model_name_or_path': str(model), 'template': 'qwen',
            'finetuning_type': 'lora', 'infer_backend': 'huggingface',
            'infer_dtype': 'bfloat16', 'do_sample': False, 'max_new_tokens': 256}
    if mode == 'lora':
        args['adapter_name_or_path'] = str(Path(adapter).resolve())
    elif mode != 'base':
        raise ValueError('mode must be base or lora')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('base', 'lora'), required=True)
    parser.add_argument('--model', default='/mnt/workspace/models/Qwen2-7B-Instruct')
    parser.add_argument('--adapter', type=Path, default=ROOT / 'outputs/qwen2_medical_smoke_v1')
    parser.add_argument('--output', type=Path, help='Default: unique timestamped file in outputs/inference_smoke')
    parser.add_argument('--dry-run', action='store_true', help='Preview only; no GPU/model validation')
    args = parser.parse_args()
    model_args = build_model_args(args.mode, args.model, args.adapter)
    if args.dry_run:
        print(json.dumps({'model_args': model_args, 'question': QUESTION, 'system': SYSTEM}, ensure_ascii=False, indent=2))
        print('DRY RUN: no model loaded or files written.')
        return
    if not (Path(args.model) / 'config.json').is_file():
        raise FileNotFoundError(f'Model config missing: {args.model}')
    adapter_hash = None
    if args.mode == 'lora':
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = args.adapter / name
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f'Adapter file missing/empty: {path}')
        adapter_hash = hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'outputs/inference_smoke' / f'{args.mode}_{stamp}.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This smoke configuration requires a CUDA GPU with BF16 support')
    from llamafactory.chat import ChatModel
    print(f'Loading {args.mode}: {args.model}', flush=True)
    if args.mode == 'lora':
        print(f'Loading adapter: {args.adapter}', flush=True)
    model = ChatModel(model_args)
    responses = model.chat([{'role': 'user', 'content': QUESTION}], system=SYSTEM)
    if not responses or not responses[0].response_text.strip():
        raise RuntimeError('Generation returned an empty response')
    response = responses[0]
    result = {'purpose': 'generation_smoke_not_quality_evaluation',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'model_args': model_args,
              'adapter_weights_sha256': adapter_hash,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'question': QUESTION, 'system': SYSTEM,
              'answer': response.response_text,
              'finish_reason': response.finish_reason,
              'response_tokens': response.response_length}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('\nQUESTION:', QUESTION)
    print('\nANSWER:', response.response_text)
    print('\nFinish reason:', response.finish_reason)
    print('Saved:', output)
    print('Generation succeeded. This is not evidence of medical quality improvement.')


if __name__ == '__main__':
    main()
