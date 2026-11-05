"""Conservative lexical checks and a restricted sodium/salt ratio verifier.
These are auditable alerts/proofs of limited scope, not clinical or semantic validation.
"""
import re
from decimal import Decimal

NUM=r'\d+(?:\.\d+)?'
QUANTITY=re.compile(r'('+NUM+r')\s*(毫克|微克|千克|克|毫升|升|周|岁|小时|分钟)')
OPS={'低于':'lt','少于':'lt','小于':'lt','不超过':'le','至多':'le','最多':'le','不高于':'le','高于':'gt','超过':'gt','大于':'gt','至少':'ge','不少于':'ge','不低于':'ge'}
BOUNDS=re.compile('('+'|'.join(sorted(OPS,key=len,reverse=True))+r')\s*('+NUM+r')\s*(毫克|微克|千克|克|毫升|升|周|岁|小时|分钟)')
RELATION=re.compile('('+NUM+r')\s*克食盐约对应\s*('+NUM+r')\s*毫克钠')
CALC=re.compile(r'(?:包装上写着)?('+NUM+r')\s*毫克钠[，,]?(?:换算成食盐)?(?:大约|约)?(?:等于|对应|是)\s*('+NUM+r')\s*克(?:食盐)?[。]?')

def split_claim(text):
    # Deliberately limited to Chinese sentence punctuation/newlines; decimal dots stay intact.
    return [s.strip() for s in re.findall(r'[^。！？\n]+[。！？]?|[^。！？\n]+$',text) if s.strip()]

def ratio_proof(text,source):
    match=CALC.fullmatch(text.strip())
    relations=[]
    for clause in re.split(r'[，,。；;\n]',source):
        clause=clause.strip()
        if RELATION.search(clause):
            relation=RELATION.fullmatch(clause)
            if relation is None:return None
            relations.append(relation)
    if not match or not relations:return None
    ratios=[]
    for r in relations:
        salt,sodium=map(Decimal,r.groups())
        if salt<=0 or sodium<=0:return None
        ratios.append(sodium/salt)
    if len(set(ratios))!=1:return None
    sodium,salt=map(Decimal,match.groups())
    if salt<0 or sodium<0 or salt*ratios[0]!=sodium:return None
    return {'scope':'restricted_ratio_arithmetic_only','sodium_mg':str(sodium),'salt_g':str(salt),'mg_per_g':str(ratios[0]),'equation':f'{sodium} / {ratios[0]} = {salt}', 'relations':[r.group(0) for r in relations]}

def inspect_claim(claim,evidence,question=''):
    text=claim['text'];ids=claim['citations']
    sources={e['citation']:e['text'] for e in evidence}
    issues=[]
    if not ids or any(type(i) is not int or i not in sources for i in ids):
        return {'issues':['unknown_citation'],'support':'unverified','sentences':split_claim(text)}
    selected=[sources[i] for i in ids]
    source='\n'.join(selected)
    sentences=split_claim(text)
    if len(sentences)!=1:issues.append('not_single_sentence')
    if re.search(r'\[\d+\]',text):issues.append('embedded_citation')
    proof=ratio_proof(text,source)
    source_quantities={(Decimal(n),u) for n,u in QUANTITY.findall(source)}
    if not proof and any((Decimal(n),u) not in source_quantities for n,u in QUANTITY.findall(text)):
        issues.append('quantity_not_in_evidence')
    bounds={}
    for op,n,u in BOUNDS.findall(source):bounds.setdefault((Decimal(n),u),set()).add(OPS[op])
    for op,n,u in BOUNDS.findall(text):
        expected=bounds.get((Decimal(n),u))
        if expected and expected!={OPS[op]}:issues.append('comparison_changed')
    # Only explicit adult/child lexical conflicts, not a general population classifier.
    if '儿童' in question and '成年' in text and '儿童' not in source:issues.append('population_mismatch')
    if '成年' in question and '儿童' in text and '成年' not in source:issues.append('population_mismatch')
    exact=any(text.strip().rstrip('。！？') in s for s in selected)
    support='restricted_arithmetic_only' if proof else ('exact_text_only' if exact else 'semantic_review_needed')
    return dict(issues=sorted(set(issues)),support=support,arithmetic=proof,sentences=sentences,citations=ids,
                limitation='Passing checks does not establish relevance, completeness, or clinical correctness. Only listed units/comparator words and explicit adult/child conflicts are covered.')
