"""Compile assistant-authored source-grounded candidates; never starts training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.prepare_evidence_sft import SYSTEM, instruction, validate_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_batch(spec, manifest, snapshot_root):
    sources = {s['id']: s for s in manifest['sources']}
    facts = {}
    for item in spec['facts']:
        if item['id'] in facts:
            raise ValueError('Duplicate fact ID')
        source = sources[item['source_id']]
        if source['split'] not in ('train', 'development'):
            raise ValueError('Reserved source cannot enter candidate authoring')
        path = snapshot_root / source['clean_relative_path']
        if sha(path) != source['clean_sha256']:
            raise ValueError('Source snapshot changed')
        body = path.read_text(encoding='utf-8')
        quote = item['quote']
        if not quote or body.count(quote) != 1:
            raise ValueError('Support must occur exactly once in snapshot')
        start = body.index(quote)
        end = start + len(quote)
        if (start and body[start-1] not in '\n。！？') or quote[-1] not in '。！？':
            raise ValueError('Support must use whole sentences; review qualifiers')
        headings = re.findall(r'^#{2,3}\s+(.+)$', body[:start], re.M)
        facts[item['id']] = dict(item, start_char=start, end_char=end,
            offset_unit='Unicode code points in UTF-8 decoded snapshot',
            source_group=source['source_group'], source=source['url'],
            title=source['title'], section=headings[-1] if headings else '正文',
            snapshot_path=source['clean_relative_path'], snapshot_sha256=source['clean_sha256'],
            split='train' if source['split']=='train' else 'validation')

    rows = []
    for example in spec['examples']:
        selected = [facts[i] for i in example['evidence_ids']]
        if not selected or len({f['split'] for f in selected}) != 1:
            raise ValueError('Empty evidence or cross-split distractor')
        if len(set(example['evidence_ids'])) != len(selected):
            raise ValueError('Duplicate evidence ID')
        evidence = [dict(citation=i, source_id=f['id'], title=f['title'],
            section=f['section'], text=f['quote'], source=f['source'],
            version='sha256:'+f['snapshot_sha256'], source_group=f['source_group'],
            start_char=f['start_char'], end_char=f['end_char']) for i,f in enumerate(selected,1)]
        support = []
        for fid in example['support_ids']:
            if fid not in example['evidence_ids']:
                raise ValueError('Support not in supplied evidence')
            support.append(dict(citation=example['evidence_ids'].index(fid)+1, quote=facts[fid]['quote']))
        rows.append(dict(id=example['id'], split=selected[0]['split'],
            kind=example['kind'], question=example['question'], answer=example['answer'],
            evidence=evidence, supports=support, missing=example['missing'],
            provenance=dict(author='assistant-authored; independent review pending',
                source_groups=sorted({f['source_group'] for f in selected}),
                fact_ids=example['evidence_ids'], candidate_batch=spec.get('version','fixture'),
                review_status='not_expert_reviewed',
                refusal_scope='only the evidence provided to this example, not the entire source article')))
    validate_rows(rows)
    return rows, list(facts.values())


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=Path('experiments/evidence_sft_v2/batch1.json'))
    parser.add_argument('--snapshots',type=Path,default=Path('data/source_snapshots/evidence_sources_v2_clean'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest_path = args.snapshots/'manifest.json'
    rows,facts = build_batch(json.loads(args.spec.read_text(encoding='utf-8')),
        json.loads(manifest_path.read_text(encoding='utf-8')), args.snapshots)
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,items in [('records.jsonl',rows),('fact_units.jsonl',facts)]:
        (args.output/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8',newline='\n')
    files={}; info={}
    for split in ('train','validation'):
        selected=[dict(system=SYSTEM,instruction=instruction(r),input='',output=r['answer']) for r in rows if r['split']==split]
        target=args.output/f'{split}.jsonl'
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8',newline='\n')
        files[target.name]=dict(count=len(selected),sha256=sha(target))
        info[f'evidence_candidate_{split}']=dict(file_name=target.name,formatting='alpaca',
            columns=dict(system='system',prompt='instruction',query='input',response='output'))
    write_json(args.output/'dataset_info.json',info)
    report=dict(status='draft_candidates_not_training_ready',**validate_rows(rows),
        facts=len(facts),files=files,spec_sha256=sha(args.spec),source_manifest_sha256=sha(manifest_path),
        reserved_test_read=False,training_started=False,
        split_kinds={s:dict(Counter(r['kind'] for r in rows if r['split']==s)) for s in ('train','validation')},
        limitations=['Assistant-authored, not independent or expert gold labels.',
            'Small batch reuses fact units; sample count is not independent-fact count.',
            'Development is for selection, not final test; semantic overlap still needs review.',
            'Whole-sentence extraction and citation validation cannot prove factual entailment.'])
    write_json(args.output/'manifest.json',report)
    review=['# 候选样本逐条审阅','', '仅评估给定片段是否支持目标答案；不是医疗诊断正确率。', '']
    for r in rows:
        review += [f"## {r['id']} / {r['split']} / {r['kind']}", '',r['question'],'',r['answer'],'',
            '未提供信息：'+(r['missing'] or '无'), '']
        for e in r['evidence']:
            review += [f"[{e['citation']}] {e['text']}",f"来源：{e['source']}；位置：{e['start_char']}:{e['end_char']}；{e['version']}",'']
        review += ['- [ ] 独立复核：回答受原文支持，条件/数值/人群未丢失，拒答范围正确。','']
    (args.output/'review.md').write_text('\n'.join(review),encoding='utf-8',newline='\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
