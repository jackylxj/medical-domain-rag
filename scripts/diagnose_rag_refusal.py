"""Six controlled generation cases; not a new retrieval or a quality benchmark."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from scripts.generate_rag_demo import prepare_prompt, SYSTEM
from scripts.infer_smoke import build_model_args
from scripts.validate_data import sha256

ROOT=Path(__file__).resolve().parents[1]
TARGET='冷冻蔬菜就一定不适合健康饮食吗？'
NEUTRAL='资料对选择冷冻蔬菜有哪些说明？'
CONTROL='资料给出的哮喘吸入药商品名和每次剂量是多少？'


def build_cases(source):
    def find(question):
        rows=[r for r in source if r.get('question')==question]
        if len(rows)!=1 or len(rows[0].get('hits',[]))<3:
            raise ValueError(f'Need exactly one saved question with Top-3: {question}')
        return rows[0]
    target,control=find(TARGET),find(CONTROL)
    if not target['hits'][0]['source_id'].startswith('diet/蔬果选择/'):
        raise ValueError('Target first hit changed; review before diagnosing Top-1')
    specifications=[
        ('A','原问法 + Top-3',TARGET,target,3),
        ('B','原问法 + Top-1',TARGET,target,1),
        ('C','中性问法 + 同一Top-1',NEUTRAL,target,1),
        ('D','中性问法 + 同一Top-3',NEUTRAL,target,3),
        ('E','资料外问题 + Top-3',CONTROL,control,3),
        ('F','资料外问题 + Top-1',CONTROL,control,1)]
    rows,cases=[],[]
    for case_id,label,question,row,k in specifications:
        rows.append(dict(question=question,hits=deepcopy(row['hits'][:k])))
        cases.append(dict(id=case_id,label=label))
    return rows,cases


def report(output):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    retrieval=output/'results.json'
    rows=json.loads(retrieval.read_text(encoding='utf-8'))
    answers=[json.loads(line) for line in (output/'generation/answers.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    config=json.loads((output/'generation/config.json').read_text(encoding='utf-8'))
    expected_args=build_model_args('base',manifest['model'],None)
    expected_args['max_new_tokens']=512
    if (config['mode']!='base' or config['prompt_version']!='v2' or not config['rag']
        or config['model_args']!=expected_args or config['system']!=SYSTEM
        or config['retrieval_file_sha256']!=sha256(retrieval)):
        raise ValueError('Generation configuration or input differs from diagnostic plan')
    if len(rows)!=6 or len(answers)!=6 or len(manifest['cases'])!=6:
        raise ValueError('Expected six completed responses')
    lines=['诊断实验：base Qwen2 + v2；同一批已保存证据，无重新检索。',
           '中性问法改变了问句表达和语用，结果只能说明问法敏感性，不能归因于某个词。']
    for case,row,answer in zip(manifest['cases'],rows,answers,strict=True):
        prompt,evidence=prepare_prompt(row,True,'v2')
        if answer['question']!=row['question'] or answer['prompt']!=prompt or answer['evidence']!=evidence:
            raise ValueError('Answer input mismatch')
        lines.extend([f'\n===== {case["id"]}: {case["label"]} =====',row['question'],
                      '回答：'+answer['answer'], '结束原因：'+str(answer['finish_reason'])])
        lines.extend(f'[{e["citation"]}] {e["text"]}' for e in evidence)
    text='\n'.join(lines)
    (output/'report.txt').write_text(text,encoding='utf-8')
    print(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/rag_corpus_v2_validation/retrieval/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rag_refusal_diagnostic_v1')
    parser.add_argument('--model',default='/mnt/workspace/models/Qwen2-7B-Instruct')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare-only',action='store_true')
    actions.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.report_only:
        report(output)
        return
    if output.exists():
        raise FileExistsError('Use a new output directory; existing diagnostic records are preserved')
    rows,cases=build_cases(json.loads(args.source.read_text(encoding='utf-8')))
    for row in rows:prepare_prompt(row,True,'v2')
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(cases=cases,model=args.model,source_sha256=sha256(args.source),
                  purpose='Context/wording diagnostic, selected known failure; no quality score',
                  comparisons=['A vs B: context count','B vs C: wording','A vs D: wording',
                               'D vs C: context count','E vs F: unanswerable control'])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.prepare_only:
        print('Prepared six cases; no model loaded:',output)
        return
    subprocess.run([sys.executable,'-m','scripts.generate_rag_demo','--mode','base','--prompt-version','v2',
                    '--model',args.model,'--retrieval',str(output/'results.json'),
                    '--output',str(output/'generation')],cwd=ROOT,check=True)
    report(output)


if __name__=='__main__':main()
