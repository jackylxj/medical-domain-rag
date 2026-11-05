"""Sequential uncached web smoke benchmark; not a throughput benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    questions = [
        '按照现有资料，成年人全天食盐摄入建议是多少？',
        '按照现有资料，成年人全天食盐摄入建议是多少？',
        '没有头痛为什么仍需要实际测量血压？',
        '家务和步行上班算不算身体活动？',
        '资料给出的哮喘吸入药商品名和每次剂量是多少？',
    ]
    records = []
    for question in questions:
        started = time.monotonic()
        request = Request(args.url+'/api/jobs', data=json.dumps({'question':question}).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=15) as response:
            job = json.load(response)
        while True:
            with urlopen(args.url+'/api/jobs/'+job['id'], timeout=15) as response:
                result = json.load(response)
            if result['status'] in ('completed','failed'):
                break
            if time.monotonic()-started > 240:
                raise TimeoutError(job['id'])
            time.sleep(0.2)
        result['client_observed_seconds'] = time.monotonic()-started
        records.append(result)
        (args.output/'jobs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        if result['status'] != 'completed':
            raise RuntimeError(f"Job failed: {job['id']}")
        print(json.dumps(dict(question=question,elapsed=result['elapsed_seconds'],timings=result['timings'],
                              answer=result['result']['answer']),ensure_ascii=False),flush=True)
    values = [r['elapsed_seconds'] for r in records]
    summary = dict(count=len(records),first_request_seconds=values[0],
                   server_task_seconds=values,mean_seconds=statistics.mean(values),
                   median_seconds=statistics.median(values),
                   notes='Five sequential requests including one repeat; no application answer cache. Includes task retrieval and generation; excludes service startup, browser polling and external network. Not a concurrent benchmark or medical accuracy evaluation.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
