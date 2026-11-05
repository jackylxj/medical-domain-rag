"""Replay rule checks over saved drafts; makes no model calls and changes no web behavior."""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
from scripts.evidence_rules import inspect_claim

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();rows=[json.loads(s) for s in args.input.read_text(encoding='utf-8').splitlines() if s.strip()]
 args.output.mkdir(parents=True,exist_ok=False)
 counts=Counter();records=[]
 for row in rows:
  checks=[]
  for i,claim in enumerate((row.get('draft') or {}).get('claims',[])):
   result=inspect_claim(claim,row['evidence'],row['question']);result['claim_index']=i
   verdict=next((v.get('supported') for v in (row.get('review') or {}).get('claims',[]) if v['index']==i),None)
   result['model_supported']=verdict
   result['review_disagrees_with_narrow_proof']=verdict is False and result['support'] in ('restricted_arithmetic_only','exact_text_only')
   counts.update(result['issues']);checks.append(result)
  record=dict(id=row['id'],question=row['question'],original_status=row['status'],checks=checks,
              disposition='flag_for_review' if any(c['issues'] or c['review_disagrees_with_narrow_proof'] for c in checks) else 'no_new_rule_alert',
              note='No replacement answer emitted. Narrow proof cannot establish full question relevance.')
  records.append(record)
 (args.output/'checks.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 summary=dict(count=len(rows),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),rule_file_sha256=hashlib.sha256(Path(__file__).with_name('evidence_rules.py').read_bytes()).hexdigest(),alerts=dict(counts),flagged_ids=[r['id'] for r in records if r['disposition']=='flag_for_review'],new_model_requests=0,warning='Known development drafts; rule matching is not medical accuracy or a complete semantic judge.')
 (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 for r in records:
  if r['disposition']=='flag_for_review':print(r['id'],json.dumps(r['checks'],ensure_ascii=False))
if __name__=='__main__':main()
