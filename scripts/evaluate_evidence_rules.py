"""Evaluate frozen rule alerts against pre-labelled synthetic cases, no model requests."""
import argparse,json,hashlib
from pathlib import Path
from scripts.evidence_rules import inspect_claim

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cases',type=Path,default=Path('experiments/evidence_rules_validation/cases.json'))
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 data=json.loads(args.cases.read_text(encoding='utf-8-sig'));cases=data['cases']
 assert len({r['id'] for r in cases})==len(cases)
 assert all(r['label'] in ('supported','unsupported') for r in cases)
 rules=Path(__file__).with_name('evidence_rules.py');before=sha(rules)
 args.output.mkdir(parents=True,exist_ok=False)
 counts=dict(tp=0,fn=0,fp=0,tn=0);records=[]
 for case in cases:
  result=inspect_claim(case['claim'],case['evidence'],case['question'])
  positive=bool(result['issues']);bad=case['label']=='unsupported'
  outcome=('tp' if positive else 'fn') if bad else ('fp' if positive else 'tn')
  counts[outcome]+=1
  records.append(dict(**case,result=result,outcome=outcome))
  print(case['id'],outcome,','.join(result['issues']) or 'no alert',result['support'])
 assert sha(rules)==before
 summary=dict(count=len(cases),counts=counts,unsupported_alert_recall=counts['tp']/(counts['tp']+counts['fn']),supported_false_alert_rate=counts['fp']/(counts['fp']+counts['tn']),cases_sha256=sha(args.cases),rules_sha256=before,new_model_requests=0,warning='Synthetic adversarial rule cases, authored with knowledge of rule scope. No alert is not approval. These rates are not medical accuracy or generalization estimates.')
 (args.output/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
