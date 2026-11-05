"""Read-only CPU audit: sampled chat-template lengths and recorded loss history; no training."""
import json,random,statistics
from pathlib import Path
from transformers import AutoTokenizer
root=Path('/mnt/workspace/RAGDemo')
rows=[json.loads(s) for s in (root/'data/processed/medical_v1/train_30000.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
sample=random.Random(20261105).sample(rows,1000)
tok=AutoTokenizer.from_pretrained('/mnt/workspace/models/Qwen2-7B-Instruct',local_files_only=True)
lengths=[len(tok.encode(tok.apply_chat_template([{'role':'user','content':r['question']},{'role':'assistant','content':r['answer']}],tokenize=False),add_special_tokens=False)) for r in sample]
assert statistics.median(lengths)>10, 'Unexpected token count; inspect tokenizer return type'
logs=[json.loads(s) for s in (root/'outputs/qwen2_medical_30k_v1/trainer_log.jsonl').read_text().splitlines() if s.strip()]
loss=[r for r in logs if 'loss' in r]
result=dict(sample_size=1000,seed=20261105,method='Model tokenizer chat template without explicit system; approximate diagnostic, not replay of LLaMA-Factory preprocessing',length_median=statistics.median(lengths),length_p95=sorted(lengths)[949],above512=sum(n>512 for n in lengths),above1024=sum(n>1024 for n in lengths),above2048=sum(n>2048 for n in lengths),first_loss_records=loss[:3],last_loss_records=loss[-3:],eval_records=[r for r in logs if 'eval_loss' in r])
print(json.dumps(result,ensure_ascii=False,indent=2))
