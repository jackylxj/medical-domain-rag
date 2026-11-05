"""Use a resident vLLM server with saved RAG evidence; no local GPU model loading."""
import argparse
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from scripts.generate_rag_demo import SYSTEM, prepare_prompt
from scripts.validate_data import sha256


def request_answer(base_url, model, prompt):
    payload=dict(model=model,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=prompt)],
                 temperature=0,top_p=1,max_tokens=512,seed=42,repetition_penalty=1.0)
    request=Request(base_url.rstrip('/')+'/chat/completions',
                    data=json.dumps(payload,ensure_ascii=False).encode(),
                    headers={'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with urlopen(request,timeout=180) as response:
            data=json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(f'vLLM HTTP {error.code}: {error.read().decode(errors="replace")[:2000]}') from error
    choices=data.get('choices')
    if not choices or not isinstance(choices[0].get('message',{}).get('content'),str) or not choices[0]['message']['content'].strip():
        raise ValueError('vLLM returned no nonempty answer')
    return data,time.monotonic()-started


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:8001/v1')
    parser.add_argument('--model',default='medical-qwen2')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    rows=json.loads(args.retrieval.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not rows:raise ValueError('Expected nonempty retrieval results')
    prompts=[prepare_prompt(row,True,'v2') for row in rows]
    if args.dry_run:
        print(json.dumps(dict(backend='vllm',questions=len(rows),model=args.model,
                             system=SYSTEM,first_prompt=prompts[0][0]),ensure_ascii=False,indent=2))
        return
    if args.output.exists():raise FileExistsError('Use a new output directory')
    with urlopen(args.base_url.rstrip('/')+'/models',timeout=10) as response:
        models=json.loads(response.read())
    if not any(m.get('id')==args.model for m in models.get('data',[])):
        raise ValueError('Requested model is not listed by server')
    args.output.mkdir(parents=True)
    config=dict(backend='vllm',mode='base',rag=True,prompt_version='v2',system=SYSTEM,
                model=args.model,base_url=args.base_url,server_models=models,
                retrieval_file_sha256=sha256(args.retrieval),temperature=0,top_p=1,
                max_tokens=512,seed=42,repetition_penalty=1.0,
                warning='Backend/template/default changes can alter answers; not a bit-identical HF comparison.')
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    limited=cited=invalid=0
    times=[]
    with (args.output/'answers.jsonl').open('x',encoding='utf-8') as stream:
        for i,(row,(prompt,evidence)) in enumerate(zip(rows,prompts,strict=True),1):
            response,elapsed=request_answer(args.base_url,args.model,prompt)
            choice=response['choices'][0]
            answer=choice['message']['content']
            numbers=sorted({int(n) for n in re.findall(r'\[(\d+)\]',answer)})
            bad=[n for n in numbers if n not in {e['citation'] for e in evidence}]
            result=dict(question=row['question'],answer=answer,evidence=evidence,prompt=prompt,
                        mode='base',backend='vllm',prompt_version='v2',cited_numbers=numbers,
                        invalid_citation_numbers=bad,finish_reason=choice.get('finish_reason'),
                        usage=response.get('usage'),system_fingerprint=response.get('system_fingerprint'),
                        request_seconds=elapsed)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            limited+=choice.get('finish_reason')=='length';cited+=bool(numbers);invalid+=bool(bad)
            times.append(elapsed)
            print(f'{i}/{len(rows)}: {elapsed:.2f}s',flush=True)
    summary=dict(count=len(rows),backend='vllm',mode='base',prompt_version='v2',rag=True,empty=0,
                 length_limited=limited,answers_with_citation_markers=cited,
                 answers_with_invalid_citation_numbers=invalid,
                 mean_request_seconds=sum(times)/len(times),
                 warning='Request timing excludes retrieval and server startup; no medical accuracy assessment.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
