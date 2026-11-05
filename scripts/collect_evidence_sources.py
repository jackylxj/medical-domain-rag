"""Collect private source snapshots via Jina; freeze source splits before QA authoring."""
import argparse,json,hashlib,re,subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,unquote

def source_group(url):
 p=urlparse(url)
 if p.scheme!='https' or p.hostname!='www.who.int' or '/news-room/fact-sheets/detail/' not in p.path:raise ValueError('Expected official WHO fact sheet URL')
 return 'who:'+unquote(p.path.rstrip('/').split('/')[-1])

def validate_registry(data):
 groups={};ids=set()
 for r in data['sources']:
  group=source_group(r['url'])
  if group in groups or r['id'] in ids:raise ValueError('Duplicate source group/ID, including language aliases')
  if group.removeprefix('who:') in data['old_development_groups']:raise ValueError('Previously used source')
  if r['split'] not in ('train','development','test_reserved'):raise ValueError('Unknown split')
  groups[group]=r['split'];ids.add(r['id'])
 return groups

def digest(b):return hashlib.sha256(b).hexdigest()

def fetch(row,root):
 folder=root/row['split'];folder.mkdir(exist_ok=True)
 raw=folder/(row['id']+'.reader.txt')
 proc=subprocess.run(['curl.exe' if __import__('os').name=='nt' else 'curl','-fLsS','--retry','1','--connect-timeout','15','--max-time','60','https://r.jina.ai/'+row['url']],capture_output=True)
 if proc.returncode:raise RuntimeError(row['id']+': reader download failed: '+proc.stderr.decode(errors='replace')[:200])
 text=proc.stdout.decode('utf-8')
 if 'Markdown Content:' not in text:raise ValueError(row['id']+': no reader content')
 body=text.split('Markdown Content:',1)[1].strip()
 if len(re.findall(r'[\u4e00-\u9fff]',body))<200:raise ValueError(row['id']+': suspiciously short Chinese article')
 title=re.search(r'^Title: (.+)',text,re.M)
 url=re.search(r'^URL Source: (.+)',text,re.M)
 if not title or not url or source_group(url[1].strip())!=source_group(row['url']):raise ValueError('Unexpected reader source')
 raw.write_bytes(proc.stdout)
 content=folder/(row['id']+'.md');content.write_text(body,encoding='utf-8',newline='\n')
 record=dict(**row,actual_title=title[1],snapshot_relative_path=content.relative_to(root).as_posix(),reader_sha256=digest(raw.read_bytes()),content_sha256=digest(content.read_bytes()),characters=len(body),paragraphs=sum(bool(x.strip()) for x in body.split('\n\n')),fetched_utc=datetime.now(timezone.utc).isoformat(),page_published_date=None,page_date_status='not_reliably_extracted; fetched time is not publication date',snapshot_version='sha256:'+digest(content.read_bytes()),clinical_review='pending',QA_status='not_authored')
 return record,body

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,default=Path('experiments/evidence_sources_v2/registry.json'));p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();raw=a.registry.read_bytes();data=json.loads(raw);validate_registry(data)
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'registry.lock.json').write_bytes(raw)
 with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda r:fetch(r,a.output),data['sources']))
 owners={};overlap=[];seen=set()
 for r,body in results:
  for paragraph in body.split('\n\n'):
   norm=re.sub(r'\s+','',paragraph)
   if len(norm)<80:continue
   h=digest(norm.encode())
   for prior in owners.get(h,[]):
    if prior['split']!=r['split'] and (h,prior['id'],r['id']) not in seen:
     overlap.append(dict(paragraph_sha256=h,left=prior['id'],right=r['id'],characters=len(norm)));seen.add((h,prior['id'],r['id']))
   owners.setdefault(h,[]).append(r)
 records=[r for r,_ in results]
 for r in records:print(r['id'],r['split'],r['characters'],'characters')
 manifest=dict(registry_sha256=digest(raw),sources=records,counts={s:sum(r['split']==s for r in records) for s in ('train','development','test_reserved')},cross_split_exact_paragraphs=overlap,status='collected_pending_semantic_dedup_and_annotation',limitations=['Source snapshots are private and excluded from public source packaging.','Exact long-paragraph matching is not semantic deduplication.','Test sources are reserved, not an authored 200-question test set.','Snapshot hashes, not inferred publication dates, identify versions.'])
 (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
 print('Exact cross-split long-paragraph matches:',len(overlap))
if __name__=='__main__':main()
