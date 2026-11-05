"""Run fresh retrieval then base-model RAG in two isolated Python environments."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--query')
    source.add_argument('--questions', type=Path)
    parser.add_argument('--retrieval-python', default='/mnt/workspace/envs/rag-retrieval-v1/bin/python')
    parser.add_argument('--model-dir', default='/mnt/workspace/models/bge-large-zh-v1.5')
    parser.add_argument('--model', default='/mnt/workspace/models/Qwen2-7B-Instruct')
    parser.add_argument('--db', type=Path, default=ROOT / 'outputs/retrieval_v1/knowledge.db')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    retrieval = [args.retrieval_python, '-m', 'scripts.retrieve_questions',
                 '--model-dir', args.model_dir, '--db', str(args.db.resolve()),
                 '--output', str(output / 'retrieval')]
    retrieval += ['--questions', str(args.questions.resolve())] if args.questions else ['--query', args.query]
    generation = [sys.executable, '-m', 'scripts.generate_rag_demo', '--mode', 'base',
                  '--prompt-version', 'v2', '--model', args.model,
                  '--retrieval', str(output / 'retrieval/results.json'),
                  '--output', str(output / 'generation')]
    for command in (retrieval, generation):
        print(command, flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        print(f'Answers: {output / "generation/answers.jsonl"}')


if __name__ == '__main__':
    main()
