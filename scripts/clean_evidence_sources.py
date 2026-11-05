"""Preserve reader snapshots; derive a versioned body excluding site/WHO-response boilerplate."""
import argparse,json,re,hashlib
from pathlib import Path
from scripts.collect_evidence_sources import digest

def clean_body(text):
 start=re.search(r'^## 重要事实\s*$',text,re.M)
 end=re.search(r'^## 世卫组织[^\n]*$',text,re.M)
 if not start or not end or end.start()<=start.start():raise ValueError('Missing/unordered article boundaries; needs manual extraction')
 return text[start.start():end.start()].strip()

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.input/'manifest.json').read_text(encoding='utf-8'))
 a.output.mkdir(parents=True,exist_ok=False)
 records=[];paragraphs={};overlap=[]
 for source in manifest['sources']:
  raw=(a.input/source['snapshot_relative_path']).read_bytes()
  if digest(raw)!=source['content_sha256']:raise ValueError('Input snapshot changed')
  clean=clean_body(raw.decode('utf-8'))
  folder=a.output/source['split'];folder.mkdir(exist_ok=True)
  path=folder/(source['id']+'.md');path.write_text(clean,encoding='utf-8',newline='\n')
  record=dict(source);record.update(clean_relative_path=path.relative_to(a.output).as_posix(),clean_sha256=digest(path.read_bytes()),clean_characters=len(clean),cleaning='From important facts heading to before WHO-response heading; includes facts and medical overview, excludes organisational response and footer.')
  for para in clean.split('\n\n'):
   norm=re.sub(r'\s+','',para)
   if len(norm)<80:continue
   key=digest(norm.encode())
   for prior in paragraphs.get(key,[]):
    if prior['split']!=record['split']:overlap.append(dict(left=prior['id'],right=record['id'],paragraph_sha256=key))
   paragraphs.setdefault(key,[]).append(record)
  records.append(record)
 result=dict(registry_sha256=manifest['registry_sha256'],raw_manifest_sha256=digest((a.input/'manifest.json').read_bytes()),counts=manifest['counts'],sources=records,cross_split_exact_paragraphs=overlap,status='source_split_frozen_no_QA_authored',limitations=['Only exact long paragraph matching completed; semantic overlap still needs review.','Page dates not asserted; version is snapshot hash.','No medical expert review; not a complete training/test dataset.'])
 (a.output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
 print(json.dumps(dict(counts=result['counts'],characters=sum(r['clean_characters'] for r in records),cross_split_exact_paragraphs=len(overlap)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
