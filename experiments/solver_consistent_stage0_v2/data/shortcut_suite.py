"""Pre-registered text-only and shallow structural shortcut suite.

Feature extraction never reads label, depth, split, IDs, or proof metadata.
Models are fit once on train and evaluated unchanged on every other split.
"""
from __future__ import annotations
import argparse, collections, csv, json, pathlib, re, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(HERE)]
from official_adapter import official_generator
from audit_data_v2 import parse_row, literal, rule, split_text

STRUCT_NAMES=['q_src_frequency','q_src_indegree','q_src_outdegree','q_src_degree','q_src_fact_any','q_src_fact_selected','q_rule_position','q_rule_length',
              'opp_src_frequency','opp_src_indegree','opp_src_outdegree','opp_src_degree','opp_src_fact_any','opp_src_fact_selected','opp_rule_position','opp_rule_length']
META_NAMES=['characters','tokens','negations','rules','facts','sentences']

def candidate_features(row,g,fol):
    sentences,qtext=split_text(row['text']); formulas,q=parse_row(row,g,fol);ql=literal(q,fol);selected=ql[1];target=ql[0]
    occurrences=collections.Counter();indeg=collections.Counter();outdeg=collections.Counter();facts=collections.defaultdict(set);candidates=[]
    for i,(sentence,f) in enumerate(zip(sentences,formulas)):
        r=rule(f,fol)
        if r:
            (a,an),(b,bn)=r;occurrences[a]+=1;occurrences[b]+=1;outdeg[a]+=1;indeg[b]+=1
            if b==target:candidates.append({'source':a,'neg':bn,'position':i/max(1,len(sentences)-1),'length':len(sentence.split())})
        else:
            l=literal(f,fol)
            if l:occurrences[l[0]]+=1;facts[l[0]].add(l[1])
    def describe(want_neg):
        cs=[c for c in candidates if c['neg']==want_neg]
        if not cs:return [0.0]*8
        c=sorted(cs,key=lambda x:(x['position'],x['source']))[0];s=c['source']
        return [occurrences[s],indeg[s],outdeg[s],indeg[s]+outdeg[s],float(bool(facts[s])),float(selected in facts[s]),c['position'],c['length']]
    qv=describe(ql[2]);ov=describe(not ql[2]);struct=np.asarray(qv+ov,dtype=float)
    meta=np.asarray([len(row['text']),len(row['text'].split()),len(re.findall(r'\bnot\b',row['text'],re.I)),sum(rule(f,fol) is not None for f in formulas),sum(rule(f,fol) is None for f in formulas),len(sentences)],dtype=float)
    # Label-free heuristics choose which final rule is real; a tie returns 0.
    heur={}
    names=['antecedent_frequency','last_hop_indegree','last_hop_outdegree','last_hop_local_degree','last_hop_entity_fact','last_hop_selected_fact','last_hop_later_position','last_hop_longer_sentence']
    for j,name in enumerate(names):
        a,b=qv[j],ov[j];heur[name]=int(a>b) if a!=b else 0
    heur['negation_parity']=int(not ql[2])
    return struct,meta,heur,qtext

def load(data_dir):
    return {p.stem:[json.loads(x) for x in p.open(encoding='utf-8')] for p in sorted(data_dir.glob('*.jsonl'))}

def run(data_dir,out_csv,out_json):
    started=time.time();g,_=official_generator();import fol
    rows=load(data_dir); parsed={}
    for split,items in rows.items():
        xs=[]
        for row in items:xs.append(candidate_features(row,g,fol))
        parsed[split]={'struct':np.stack([x[0] for x in xs]),'meta':np.stack([x[1] for x in xs]),'heur':[x[2] for x in xs],'query':[x[3] for x in xs],
                       'full':[r['text'] for r in items],'y':np.asarray([r['label'] for r in items]),'depth':np.asarray([r['depth'] for r in items])}
    tr=parsed['train'];majority=int(tr['y'].mean()>=.5)
    models={
      'query_tfidf_logreg':make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000),LogisticRegression(C=1.0,max_iter=1000,random_state=0)),
      'full_text_tfidf_logreg':make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000),LogisticRegression(C=1.0,max_iter=1000,random_state=0)),
      'structural_logreg':make_pipeline(__import__('sklearn').preprocessing.StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=0)),
      'structural_tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=20,random_state=0),
      'structural_random_forest':RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=0,n_jobs=-1),
      'metadata_logreg':make_pipeline(__import__('sklearn').preprocessing.StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=0)),
    }
    models['query_tfidf_logreg'].fit(tr['query'],tr['y']);models['full_text_tfidf_logreg'].fit(tr['full'],tr['y'])
    for n in ('structural_logreg','structural_tree','structural_random_forest'):models[n].fit(tr['struct'],tr['y'])
    models['metadata_logreg'].fit(tr['meta'],tr['y'])
    stumps={}
    all_names=STRUCT_NAMES+META_NAMES
    all_train=np.concatenate([tr['struct'],tr['meta']],axis=1)
    for j,name in enumerate(all_names):
        stumps[name]=DecisionTreeClassifier(max_depth=1,min_samples_leaf=20,random_state=0).fit(all_train[:,[j]],tr['y'])
    records=[]
    def add(method,category,split,depth,y,pred,threshold):
        acc=float(np.mean(np.asarray(pred)==y));records.append({'method':method,'category':category,'split':split,'depth':depth,'n':len(y),'accuracy':acc,'threshold':threshold,'pass':acc<=threshold})
    heuristic_names=['majority']+sorted(tr['heur'][0])
    for split,d in parsed.items():
        depths=['all']+sorted(set(d['depth'].tolist()))
        model_predictions={
          'query_tfidf_logreg':models['query_tfidf_logreg'].predict(d['query']),
          'full_text_tfidf_logreg':models['full_text_tfidf_logreg'].predict(d['full']),
          'structural_logreg':models['structural_logreg'].predict(d['struct']),
          'structural_tree':models['structural_tree'].predict(d['struct']),
          'structural_random_forest':models['structural_random_forest'].predict(d['struct']),
          'metadata_logreg':models['metadata_logreg'].predict(d['meta'])}
        all_x=np.concatenate([d['struct'],d['meta']],axis=1)
        stump_pred={name:stumps[name].predict(all_x[:,[j]]) for j,name in enumerate(all_names)}
        for depth in depths:
            mask=np.ones(len(d['y']),dtype=bool) if depth=='all' else d['depth']==depth;y=d['y'][mask]
            add('majority','no_training',split,depth,y,np.full(len(y),majority),.55)
            for name in heuristic_names[1:]:
                threshold=.52 if name=='antecedent_frequency' else .55
                add(name,'no_training',split,depth,y,[d['heur'][i][name] for i in np.flatnonzero(mask)],threshold)
            for name,pred in model_predictions.items():
                add(name,'shallow_classifier',split,depth,y,pred[mask],.60)
            for name,pred in stump_pred.items():add('single_'+name,'single_surface_feature',split,depth,y,pred[mask],.55)
    # Gate is applied to held-out IID/OOD only; antecedent frequency is also
    # required to stay <=52% in every reported held-out stratum.
    gated=[r for r in records if r['split']=='iid' or r['split'].startswith('ood_')]
    failures=[r for r in gated if not r['pass']]
    summary={'scope':'all rows; models fit only on train','data_dir':str(data_dir.resolve()),'started_unix':started,'ended_unix':time.time(),
             'records':len(records),'passed':not failures,'failure_count':len(failures),'failures':failures[:200],
             'max_heldout_accuracy':max(r['accuracy'] for r in gated),'max_heldout_record':max(gated,key=lambda r:r['accuracy']),
             'feature_names':{'structural':STRUCT_NAMES,'metadata':META_NAMES},'thresholds':{'no_training':.55,'antecedent_frequency':.52,'shallow_classifier':.60,'single_surface_feature':.55}}
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    out_json.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('records','passed','failure_count','max_heldout_accuracy','max_heldout_record')},indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=pathlib.Path,default=ROOT/'data/generated');p.add_argument('--output-csv',type=pathlib.Path,default=ROOT/'artifacts/data_shortcut_results.csv');p.add_argument('--output-json',type=pathlib.Path,default=ROOT/'artifacts/data_shortcut_summary.json');a=p.parse_args()
    s=run(a.data_dir,a.output_csv,a.output_json);raise SystemExit(0 if s['passed'] else 2)
