"""Generate validation predictions and character ROUGE-L; not clinical accuracy."""
import argparse
import json
from pathlib import Path
from scripts.infer_smoke import build_model_args, SYSTEM
from scripts.validate_data import records, sha256

ROOT = Path(__file__).resolve().parents[1]


def rouge_l(prediction, reference):
    # Explicit character-level LCS F1, punctuation retained, whitespace removed.
    a, b = ''.join(prediction.split()), ''.join(reference.split())
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for char in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if char == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('base', 'lora'), required=True)
    p.add_argument('--model', default='/mnt/workspace/models/Qwen2-7B-Instruct')
    p.add_argument('--adapter', type=Path, default=ROOT / 'outputs/qwen2_medical_pilot_v1')
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/medical_v1/dev100.jsonl')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a new output directory: {args.output}')
    data = list(records(args.data))
    manifest = json.loads((args.data.parent / 'manifest.json').read_text(encoding='utf-8'))
    data_hash = sha256(args.data)
    if data_hash != manifest['files'][args.data.name]['sha256']:
        raise ValueError('Evaluation data hash mismatch')
    if args.mode == 'lora' and not (args.adapter / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Adapter weights not found')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This configuration requires CUDA and BF16')
    from llamafactory.chat import ChatModel
    model_args = build_model_args(args.mode, args.model, args.adapter)
    model_args['max_new_tokens'] = 512
    model = ChatModel(model_args)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'mode': args.mode, 'model_args': model_args, 'system': SYSTEM,
                'data_sha256': data_hash, 'data_file': str(args.data),
                'adapter_sha256': sha256(args.adapter / 'adapter_model.safetensors') if args.mode == 'lora' else None,
                'purpose': 'validation_comparison_not_clinical_accuracy'}
    (args.output / 'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    scores, lengths, empty = [], 0, 0
    with (args.output / 'predictions.jsonl').open('x', encoding='utf-8') as f:
        for i, row in enumerate(data, 1):
            response = model.chat([{'role': 'user', 'content': row['question']}], system=SYSTEM)[0]
            score = rouge_l(response.response_text, row['answer'])
            scores.append(score)
            lengths += response.finish_reason == 'length'
            empty += not response.response_text.strip()
            result = {'id': row.get('id'), 'source_record_index': row.get('source_record_index'),
                      'question': row['question'], 'reference': row['answer'],
                      'prediction': response.response_text, 'finish_reason': response.finish_reason,
                      'response_tokens': response.response_length, 'rouge_l_char_f1': score}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(f'{i}/{len(data)} completed', flush=True)
    summary = {'count': len(scores), 'rouge_l_char_f1_0_to_100': 100 * sum(scores) / len(scores),
               'length_limited': lengths, 'empty': empty,
               'metric_definition': 'mean per-example character LCS F1; whitespace removed; punctuation retained',
               'warning': 'Lexical overlap, not medical accuracy; not directly comparable to other ROUGE tokenization.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
