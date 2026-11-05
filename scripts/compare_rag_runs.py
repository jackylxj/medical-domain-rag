"""Check matched inputs/settings and print prompt-version comparison; no automatic medical grade."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    args=parser.parse_args()
    configs=[json.loads((p/'config.json').read_text(encoding='utf-8')) for p in (args.before,args.after)]
    for key in ('mode','rag','model_args','system','retrieval_file_sha256','adapter_sha256'):
        if key not in configs[0] or key not in configs[1] or configs[0][key]!=configs[1][key]:
            raise ValueError(f'Comparison settings differ or missing: {key}')
    before,after=read(args.before/'answers.jsonl'),read(args.after/'answers.jsonl')
    if not before or len(before)!=len(after):
        raise ValueError('Empty or unequal response counts')
    for i,(a,b) in enumerate(zip(before,after,strict=True),1):
        if a['question']!=b['question'] or a['evidence']!=b['evidence']:
            raise ValueError(f'Question/evidence differ at row {i}')
    print('输入、证据、模型配置一致；仅提示词版本不同的实验可据此对照。')
    print('本脚本不自动判定回答正确性或医学准确率。')
    for i,(a,b) in enumerate(zip(before,after,strict=True),1):
        print(f'\n===== 问题 {i} =====\n{a["question"]}')
        print(f'【{configs[0]["prompt_version"]}】{a["answer"]}')
        print(f'【{configs[1]["prompt_version"]}】{b["answer"]}')
        for e in b['evidence']:
            print(f'[{e["citation"]}] {e["text"]}')


if __name__=='__main__':main()
