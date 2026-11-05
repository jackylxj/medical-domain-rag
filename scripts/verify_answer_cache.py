"""Real Redis and web integration checks. Run against an idle, freshly restarted web service."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from uuid import uuid4
from app.answer_cache import AnswerCache, cache_key
from app.rag_service import JobService


def ask(url,question):
    with urlopen(Request(url+'/api/jobs',data=json.dumps({'question':question}).encode(),
                         headers={'Content-Type':'application/json'}),timeout=10) as response:
        job=json.load(response)
    started=time.monotonic()
    while time.monotonic()-started < 120:
        with urlopen(url+'/api/jobs/'+job['id'],timeout=10) as response:
            result=json.load(response)
        if result['status']=='failed':
            raise RuntimeError(result)
        if result['status']=='completed':
            return result
        time.sleep(0.05)
    raise TimeoutError(job['id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for q in ['成年人全天食盐摄入建议是多少？']*2 + ['家务和步行上班算不算身体活动？']*2 + ['资料给出的哮喘吸入药商品名和每次剂量是多少？']*2:
        r=ask(args.url,q)
        rows.append(r)
        (args.output/'jobs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'id':r['id'],'cache':r['cache'],'seconds':r['elapsed_seconds'],'answer':r['result']['answer']},ensure_ascii=False),flush=True)
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['cache']['status']=='miss' and a['cache']['stored']
        assert b['cache']['status']=='hit' and b['cache']['source_job_id']==a['id']
        assert a['result']['answer']==b['result']['answer']
        assert a['result']['evidence']==b['result']['evidence']
        assert not b['result']['generation_executed'] and 'usage' not in b['result']
        assert 'generating_wall_seconds' not in b['timings']
    cache=AnswerCache.from_url('redis://127.0.0.1:6379/0',ttl=1)
    context=dict(deployment=uuid4().hex,corpus='fixture-a',model='fixture-base',prompt='fixture-v2')
    key=cache_key('TTL and invalidation fixture',context)
    entry=dict(result={'answer':'fixture'},source_job_id='fixture')
    cache.put(key,entry)
    assert cache.get(key)==entry
    for field in context:
        altered=dict(context,**{field:'different'})
        assert cache.get(cache_key('TTL and invalidation fixture',altered)) is None
    time.sleep(1.2)
    assert cache.get(key) is None
    # A closed local port simulates unavailable Redis without interrupting the live cache.
    service=JobService(args.output/'outage', '/mnt/workspace/envs/rag-retrieval-v1/bin/python',
                       '/usr/local/bin/python','/mnt/workspace/models/bge-large-zh-v1.5',
                       '/mnt/workspace/models/Qwen2-7B-Instruct',
                       '/mnt/workspace/RAGDemo/outputs/retrieval_v2/knowledge.db',
                       backend='vllm',retrieval_url='http://127.0.0.1:8002',
                       answer_cache=AnswerCache.from_url('redis://127.0.0.1:6398/0'))
    job=service.submit('没有头痛为什么仍需要实际测量血压？')
    started=time.monotonic()
    while time.monotonic()-started < 120:
        outage=service.get(job['id'])
        if outage['status'] in ('completed','failed'):
            break
        time.sleep(0.05)
    assert outage['status']=='completed' and outage['cache']['status']=='bypass_error'
    assert outage['result']['generation_executed']
    (args.output/'outage_result.json').write_text(json.dumps(outage,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(count=len(rows),miss_seconds=[r['elapsed_seconds'] for r in rows[::2]],
                 hit_seconds=[r['elapsed_seconds'] for r in rows[1::2]],
                 checks=['three exact answer/evidence matches','different question misses','no generation metrics on hits',
                         'real Redis one-second TTL expiration','synthetic version-context changes miss in real Redis',
                         'closed Redis port falls back to real retrieval and vLLM generation'],
                 note='Version changes use fixtures, not actual model replacement. Server task timing excludes browser polling/network; no concurrency or medical accuracy claim.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
