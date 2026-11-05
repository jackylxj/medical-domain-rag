"""New wording / unsupported-detail checks after the initial diagnostic; not a blind benchmark."""
import json
from pathlib import Path
import scripts.diagnose_prompt_v4 as diagnostic
from scripts.generate_rag_demo import prepare_prompt
root=Path('/mnt/workspace/RAGDemo')
rows=[json.loads(s) for s in diagnostic.SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
cases=[(1,'资料给出的对应关系中，800毫克钠约等于多少克食盐？'),(4,'在什么条件下，冷冻蔬果也可以选择？'),(5,'步行去上班属于资料所说的身体活动吗？'),(9,'资料有没有列出哮喘吸入药的具体品牌？'),(10,'仅依据资料，筛查孕期血糖应该安排在第几周？'),(0,'资料建议儿童每天最多吃多少克盐？')]
new=[]
for index,q in cases:
 row=rows[index]
 prompt,evidence=prepare_prompt({'question':q,'hits':row['evidence']},True,'v2')
 new.append(dict(question=q,prompt=prompt,evidence=evidence))
source=root/'outputs/prompt_diagnostic_v4/new_checks.jsonl'
with source.open('x',encoding='utf-8') as f:
 for row in new:f.write(json.dumps(row,ensure_ascii=False)+'\n')
diagnostic.SOURCE=source
diagnostic.OUT=root/'outputs/prompt_diagnostic_v4_followup'
diagnostic.main()
