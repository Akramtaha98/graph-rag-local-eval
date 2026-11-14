import json,re,csv,pickle,collections,itertools,sys
import numpy as np
from scipy import stats
P=sys.argv[1] if len(sys.argv)>1 else '.'
d=json.load(open(f'{P}/results/detailed_v2.json'));old=json.load(open(f'{P}/results/detailed.json'))
qs=json.load(open(f'{P}/questions.json'))
S=['naive_rag','bm25_hybrid','graph_rag']
OPEN=re.compile(r"^[^.]{0,120}?(does not|doesn't|do not|not) (contain|provide|include)[^.]{0,60}?(enough|sufficient)?[^.]*?(information|context)|^[^.]{0,80}(insufficient|not enough) (information|context)",re.I)
def sent1(a): return re.split(r"(?<=[.!?])\s",a.strip(),1)[0]
def rtype(r):
    a=r['answer']; s=sent1(a)
    if re.search(r"(does not|doesn't) contain (enough|sufficient)|not enough (information|context)|insufficient",s,re.I):
        return 'partial' if re.search(r"however|infer|inference|but we can|based on the available",a[len(s):],re.I) else 'full'
    return 'none'
rt={i:rtype(r) for i,r in enumerate(d) if r['system']!='no_rag'}
print({k:[(i,d[i]['system'],d[i]['v2_faithfulness']) for i in rt if rt[i]==k] for k in ('full','partial')})
rng=np.random.default_rng(0)
def pair(a,b,qq,excl):
    byi={s:{r['question']:(i,r) for i,r in enumerate(d) if r['system']==s} for s in S}
    xs=[]
    for q in qq:
        ia,ra=byi[a][q];ib,rb=byi[b][q]
        if ra['v2_faithfulness'] is None or rb['v2_faithfulness'] is None: continue
        if excl and (rt[ia]!='none' or rt[ib]!='none'): continue
        xs.append(ra['v2_faithfulness']-rb['v2_faithfulness'])
    x=np.array(xs);n=len(x);m=x.mean()
    ci=np.percentile(rng.choice(x,(10000,n)).mean(1),[2.5,97.5]);c=x-m
    p=float(np.mean(np.abs(rng.choice(c,(10000,n)).mean(1))>=abs(m)))
    return dict(n=n,diff=round(float(m),3),ci=[round(float(v),2) for v in ci],p=round(p,3),ties=int((x==0).sum()))
Q=[q['question'] for q in qs]
out={'refusal_idx':{k:[i for i in rt if rt[i]==k] for k in ('full','partial')}}
out['excl_means']={}
for s in S:
    v=[r['v2_faithfulness'] for i,r in enumerate(d) if r['system']==s and r['v2_faithfulness'] is not None and rt[i]=='none']
    out['excl_means'][s]=dict(n=len(v),mean=round(float(np.mean(v)),3),ci=[round(float(x),2) for x in np.percentile(rng.choice(v,(10000,len(v))).mean(1),[2.5,97.5])],n5=int(sum(x==5 for x in v)))
    out['excl_means'][s]['full']=sum(1 for i,r in enumerate(d) if r['system']==s and rt.get(i)=='full')
    out['excl_means'][s]['partial']=sum(1 for i,r in enumerate(d) if r['system']==s and rt.get(i)=='partial')
out['excl_pairs']={f'{a}|{b}':pair(a,b,Q,True) for a,b in itertools.combinations(S,2)}
out['all_pairs']={f'{a}|{b}':pair(a,b,Q,False) for a,b in itertools.combinations(S,2)}
# chunk counts per source
ch=pickle.load(open(f'{P}/data/index/chunks.pkl','rb'))
out['chunks']=collections.Counter(c['source'] for c in ch)
out['words']={}
import os
for f in os.listdir(f'{P}/data/docs'): out['words'][f]=len(open(f'{P}/data/docs/{f}',encoding='utf-8',errors='ignore').read().split())
# human sample
key={int(r['id']):r for r in csv.DictReader(open(f'{P}/results/human_eval_v2_key.csv',encoding='utf-8'))}
H={n:{int(r['id']):r for r in csv.DictReader(open(f'{P}/results/human_eval_v2_rater{"AB"[n-2]}.csv',encoding='utf-8'))} for n in (2,3)}
ids=sorted(key)
J=np.array([float(key[i]['judge_faithfulness_v2']) for i in ids]);ins=np.array([key[i]['inserted']=='1' for i in ids])
A=np.array([int(H[2][i]['human_faithfulness']) for i in ids]);B=np.array([int(H[3][i]['human_faithfulness']) for i in ids])
def mat(x,y):
    return {int(a):{int(b):int(((x==a)&(y==b)).sum()) for b in range(1,6)} for a in range(1,6)}
out['mat_judge_A']=mat(J,A);out['mat_judge_B']=mat(J,B);out['mat_A_B']=mat(A,B)
# claims for the 12 inserted bases
rows=[]
for i in ids:
    if key[i]['inserted']=='1':
        o=d[int(key[i]['orig_index'])]
        rows.append(dict(id=i,system=key[i]['system'],n_claims_base=len(o['v2_claims']),judge=float(key[i]['judge_faithfulness_v2']),A=int(H[2][i]['human_faithfulness']),B=int(H[3][i]['human_faithfulness'])))
out['inserted']=rows
out['inserted_n_ge9']=sum(r['n_claims_base']>=9 for r in rows)
out['inserted_mean_claims']=round(float(np.mean([r['n_claims_base'] for r in rows])),2)
# disagreement examples
dis=[(i,int(H[2][i]['human_faithfulness']),int(H[3][i]['human_faithfulness']),float(key[i]['judge_faithfulness_v2']),key[i]['inserted'],H[2][i]['notes'],H[3][i]['notes'],key[i]['system'])for i in ids if abs(int(H[2][i]['human_faithfulness'])-int(H[3][i]['human_faithfulness']))>=2]
out['serious_disagreements']=[dict(id=a,A=b,B=c,judge=e,inserted=f,noteA=g,noteB=h,system=k) for a,b,c,e,f,g,h,k in dis]
out['n_serious']=len(dis)
out['sig_by_inserted']={'A_on_ins':A[ins].tolist(),'B_on_ins':B[ins].tolist()}
json.dump(out,open('facts.json','w'),indent=1,default=str)
print(json.dumps({k:out[k] for k in ('excl_means','excl_pairs','chunks','words','inserted_n_ge9','inserted_mean_claims','n_serious')},indent=1,default=str))
print(out['inserted']);print(out['mat_judge_A'],out['mat_judge_B'],out['mat_A_B'])
for r in out['serious_disagreements'][:12]:print(r)
