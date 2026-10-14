"""Development-only calibration, problem-cluster bootstrap, then locked audit."""
import argparse,collections,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
METRICS=['G_local','G_local_unit','G_source','G_consumer','I','G_source_span','mask_attenuation','shallow_gap','shallow_retained','source_depth1_null','masked_shape_null','raw_local_gain']

def read(path):return json.loads(path.read_text(encoding='utf8'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def assemble(records):
    groups=collections.defaultdict(dict)
    for r in records:
        if r['condition'] in groups[r['case_id']]:raise ValueError('Duplicate case/condition '+r['case_id'])
        groups[r['case_id']][r['condition']]=r
    output=[];incomplete=[]
    for case,c in groups.items():
        if not {'11','21','12','22'}<=c.keys():incomplete.append(case);continue
        a,b,d,e=[c[k] for k in ['11','21','12','22']]
        assert len({v['text_sha256'] for v in c.values()})==1
        row={k:e[k] for k in ['case_id','sample_id','parent_id','dataset','H','kind','native_source_depth','native_consumer_depth','i','j','producer_node','consumer_node','distance_tokens','relation','renamed']}
        row.update({'matched_dependent_i':e.get('matched_dependent_i'),'G_local':a['source_next_ce_nats']-b['source_next_ce_nats'],
                    'matching_abs_local_native_ce_difference':e.get('matching_abs_local_native_ce_difference'),
                    'matching_distance_ratio':e.get('matching_distance_ratio'),
                    'G_local_unit':a['source_semantic_unit_ce_nats']-b['source_semantic_unit_ce_nats'],
                    'G_source':e['U']-d['U'],'G_consumer':d['U']-a['U'],'I':e['U']-b['U']-d['U']+a['U'],
                    'G_source_span':e['U_semantic_span']-d['U_semantic_span'],'source_depth1_null':b['U']-a['U'],
                    **{'U'+k:c[k]['U'] for k in ['11','21','12','22']},
                    'source_continue_probability':b['source_stop_scores']['1']['continue_probability'] if b.get('source_stop_scores') else None,
                    'consumer_continue_probability':e['consumer_stop_scores']['1']['continue_probability'] if e.get('consumer_stop_scores') else None})
        raw={v['depth']:v['raw_ce_nats'] for v in b['per_loop_target_and_top8_logits'] if v['position']==b['i']}
        row['raw_local_gain']=raw.get(1,0)-raw.get(2,0) if {1,2}<=raw.keys() else None
        if '22_mask' in c:
            row.update(mask_attenuation=e['U']-c['22_mask']['U'],masked_shape_null=c['22_mask']['U']-d['U'])
        if '22_shallow' in c:
            row.update(shallow_gap=e['U']-c['22_shallow']['U'],shallow_retained=c['22_shallow']['U']-d['U'])
        if '22_repeat' in c:row['repeat_difference']=c['22_repeat']['U']-e['U']
        row['semantic_anchor_gain_correctness_change']=int(e['consumer_next_token_correct'])-int(d['consumer_next_token_correct'])
        row['logical_iterations_cells']={k:c[k]['logical_full_prefix_iterations'] for k in ['11','21','12','22']}
        output.append(row)
    return output,incomplete

def estimate(rows,key,reps=5000):
    groups=collections.defaultdict(list)
    for r in rows:
        value=r.get(key)
        if value is not None and np.isfinite(value):groups[r['parent_id']].append(float(value))
    if not groups:return {'n_pairs':0,'n_clusters':0,'mean':None,'ci95':[None,None]}
    # Equal weight per original problem, not per token or pair.
    vals=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(42)
    means=vals[rng.integers(0,len(vals),size=(reps,len(vals)))].mean(axis=1)
    return {'n_pairs':sum(map(len,groups.values())),'n_clusters':len(vals),'mean':float(vals.mean()),
            'ci95':np.quantile(means,[.025,.975]).tolist(),'fraction_positive_clusters':float(np.mean(vals>0))}

def matched_comparisons(rows):
    index={(r['sample_id'],r['i'],r['j']):r for r in rows if r['kind']=='dependent'};out=[]
    for n in rows:
        if n['kind']!='nondependent':continue
        d=index.get((n['sample_id'],n['matched_dependent_i'],n['j']))
        if not d:continue
        # Same native source depth was required to match absolute 2x2 budgets.
        assert n['logical_iterations_cells']==d['logical_iterations_cells']
        out.append({'parent_id':d['parent_id'],'dataset':d['dataset'],'H':d['H'],'sample_id':d['sample_id'],
                    'native_source_depth':d['native_source_depth'],'G_local':d['G_local'],
                    'dependent_case':d['case_id'],'nondependent_case':n['case_id'],
                    'dependent_minus_nondependent':d['G_source']-n['G_source']})
    return out

def calibrate(rows):
    dep=[r for r in rows if r['kind']=='dependent'];native1=[abs(r['G_local']) for r in dep if r['native_source_depth']==1]
    repeat=max([abs(r.get('repeat_difference',0)) for r in dep] or [0])
    shape=np.array([abs(r['masked_shape_null']) for r in dep if 'masked_shape_null' in r])
    threshold=max(.01,float(np.quantile(native1,.25)) if native1 else .01,5*repeat)
    content={'split_used':'development_only','rule':'max(0.01, dev native-source-K1 Q25(abs(G_local)), 5*max(abs(repeat_difference)))',
             'local_small_threshold_nats':threshold,'native1_development_pairs':len(native1),'dev_repeat_max_abs_nats':repeat,
             'masked_shape_null_p95_abs_nats':float(np.quantile(shape,.95)) if len(shape) else None,
             'masked_shape_null_max_abs_nats':float(shape.max()) if len(shape) else None,
             'minimum_delayed_effect_nats':.01,'threshold_scope':'anchor local gain, not whole-unit local gain',
             'shape_null_scope':'diagnostic credibility check, not a post-hoc sample exclusion or local-gain threshold',
             'manifest_sha256':sha(ROOT/'outputs/sample_manifest.jsonl'),'config_sha256':sha(ROOT/'config.yaml'),
             'development_results_sha256':sha(ROOT/'outputs/intervention_results.jsonl')}
    p=ROOT/'outputs/development_thresholds.json'
    if p.exists():
        old=read(p)
        for key in ['local_small_threshold_nats','manifest_sha256','config_sha256']:assert old[key]==content[key]
        return old
    write(p,content);return content

def summarize(rows,threshold):
    report={};all_groups=[('all',rows)]
    for dataset in sorted({r['dataset'] for r in rows}):all_groups.append((dataset,[r for r in rows if r['dataset']==dataset]))
    for name,group in all_groups:
        report[name]={}
        for kind in ['dependent','nondependent']:
            subset=[r for r in group if r['kind']==kind];views={'all':subset,'native_source_K1':[r for r in subset if r['native_source_depth']==1],
                    'low_local_abs_gain':[r for r in subset if abs(r['G_local'])<=threshold],
                    'native_source_K1_and_low_local':[r for r in subset if r['native_source_depth']==1 and abs(r['G_local'])<=threshold]}
            for h in sorted({r['H'] for r in subset}):views[f'H{h}']=[r for r in subset if r['H']==h]
            report[name][kind]={view:{metric:estimate(vals,metric) for metric in METRICS+['U11','U21','U12','U22','semantic_anchor_gain_correctness_change']} for view,vals in views.items()}
    return report

def continuous(rows):
    from scipy.stats import spearmanr
    out={}
    for dataset in sorted({r['dataset'] for r in rows}):
        a=[r for r in rows if r['dataset']==dataset and r['kind']=='dependent'];groups=collections.defaultdict(list)
        for r in a:groups[r['parent_id']].append(r)
        x=np.array([np.mean([r['G_local'] for r in v]) for _,v in sorted(groups.items())]);y=np.array([np.mean([r['G_source'] for r in v]) for _,v in sorted(groups.items())])
        rho=spearmanr(x,y).statistic if len(x)>2 and x.std()>0 and y.std()>0 else np.nan
        slope=np.polyfit(x,y,1)[0] if len(x)>2 and x.std()>0 else np.nan
        out[dataset]={'n_clusters':len(x),'spearman_rho':float(rho) if np.isfinite(rho) else None,
                      'OLS_slope_descriptive_only':float(slope) if np.isfinite(slope) else None,'scope':'problem-mean association; not causal mediation or independence proof'}
    return out

def paired_depth(rows):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if r['dataset']=='controlled_symbol_chains_new_diagnostic' and r['kind']=='dependent':groups[r['parent_id']][r['H']].append(r)
    pairs=[]
    for parent,h in groups.items():
        if {4,16}<=h.keys():
            pairs.append({'parent_id':parent,**{key:float(np.mean([r[key] for r in h[16]])-np.mean([r[key] for r in h[4]])) for key in ['G_source','G_consumer','G_local','G_local_unit']}})
    return {key:estimate(pairs,key) for key in ['G_source','G_consumer','G_local','G_local_unit']}

def make_plots(rows,summary,threshold,phase):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    directory=ROOT/'outputs/figures'/phase;directory.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.6))
    for dataset,color in [('nestful_v2_math','#2563eb'),('controlled_symbol_chains_new_diagnostic','#d97706')]:
        a=[r for r in rows if r['kind']=='dependent' and r['dataset']==dataset]
        ax.scatter([r['G_local'] for r in a],[r['G_source'] for r in a],s=14,alpha=.6,label=dataset.split('_')[0],color=color)
    ax.axhline(0,color='grey',lw=.8);ax.axvline(0,color='grey',lw=.8);ax.axvspan(-threshold,threshold,color='grey',alpha=.1)
    ax.set(xlabel='Immediate source mixture-CE gain (nats)',ylabel='Later consumer mixture-CE gain (nats)',title='Fixed-text native SDPA audit');ax.legend();fig.tight_layout();fig.savefig(directory/'local_vs_delayed.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(dataset,color) in zip(axes,[('nestful_v2_math','#2563eb'),('controlled_symbol_chains_new_diagnostic','#d97706')]):
        a=[r for r in rows if r['kind']=='dependent' and r['dataset']==dataset];keys=['G_source','G_consumer','mask_attenuation','shallow_gap','masked_shape_null']
        for i,key in enumerate(keys):
            v=estimate(a,key)
            if v['mean'] is not None:
                lo,hi=v['ci95'];ax.errorbar(i,v['mean'],yerr=[[max(0,v['mean']-lo)],[max(0,hi-v['mean'])]],fmt='o',color=color,capsize=4)
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(len(keys)),['Source','Consumer','KV mask','Deep vs\nshallow','Shape null'],rotation=15);ax.set(ylabel='Gain (nats), 95% problem-cluster CI',title=dataset.split('_')[0])
    fig.tight_layout();fig.savefig(directory/'mechanism_controls.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));dataset='controlled_symbol_chains_new_diagnostic'
    for h in [4,16]:
        a=[r for r in rows if r['kind']=='dependent' and r['dataset']==dataset and r['H']==h];v=estimate(a,'G_source')
        if v['mean'] is not None:ax.errorbar(h,v['mean'],yerr=[[max(0,v['mean']-v['ci95'][0])],[max(0,v['ci95'][1]-v['mean'])]],fmt='o',capsize=5)
    ax.axhline(0,color='grey',lw=.8);ax.set(xticks=[4,16],xlabel='Dependency depth H (N=16 operations)',ylabel='Later source benefit (nats)',title='Correct intermediate states are teacher-forced');fig.tight_layout();fig.savefig(directory/'depth_effect.png');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','audit','rename','dev_aligned','audit_aligned','rename_aligned','depths_aligned'],required=True);a=p.parse_args()
    all_records=lines(ROOT/'outputs/intervention_results.jsonl');rows,incomplete=assemble([r for r in all_records if r['phase']==a.phase])
    thresholds=calibrate(rows) if a.phase=='dev' else read(ROOT/'outputs/development_thresholds.json')
    assert thresholds['manifest_sha256']==sha(ROOT/'outputs/sample_manifest.jsonl')
    assert thresholds['config_sha256']==sha(ROOT/'config.yaml')
    for r in rows:r['low_local']=abs(r['G_local'])<=thresholds['local_small_threshold_nats']
    matches=matched_comparisons(rows);summary=summarize(rows,thresholds['local_small_threshold_nats'])
    comparisons={}
    for dataset in sorted({r['dataset'] for r in rows}):
        vals=[r for r in matches if r['dataset']==dataset];comparisons[dataset]={v:estimate(s,'dependent_minus_nondependent') for v,s in [('all',vals),('native_source_K1',[r for r in vals if r['native_source_depth']==1]),('low_local',[r for r in vals if abs(r['G_local'])<=thresholds['local_small_threshold_nats']])]}
    report={'phase':a.phase,'thresholds':thresholds,'n_complete_cases':len(rows),'n_incomplete_cases':len(incomplete),'incomplete_case_ids':incomplete,
            'independent_parent_units':len({r['parent_id'] for r in rows}),'statistics':summary,'dependency_matched_comparisons':comparisons,
            'continuous_local_vs_delayed':continuous(rows),'cases':rows,'matched_cases':matches,
            'paired_H16_minus_H4':paired_depth(rows),
            'bootstrap':'5000 paired original-problem/group resamples, seed42, equal mean weight per parent; descriptive 95% CIs without multiplicity adjustment',
            'identity_max_abs_I_minus_G_source':max([abs(r['I']-r['G_source']) for r in rows] or [0]),
            'source_K2_consumer_K1_max_abs_effect':max([abs(r['source_depth1_null']) for r in rows] or [0]),
            'cache_equivalent_masked_null_max_abs_nats':max([abs(r.get('masked_shape_null',0)) for r in rows] or [0]),
            'claim_scope':'fixed-reference semantic-anchor next-token utility; not free-generation task success'}
    write(ROOT/'outputs'/f'analysis_{a.phase}.json',report)
    if a.phase.startswith('audit'):make_plots(rows,summary,thresholds['local_small_threshold_nats'],a.phase)
    print(json.dumps({k:report[k] for k in ['phase','n_complete_cases','independent_parent_units','identity_max_abs_I_minus_G_source','source_K2_consumer_K1_max_abs_effect']},indent=2))
    for dataset in sorted({r['dataset'] for r in rows}):
        v=summary[dataset]['dependent']['all'];print(dataset,json.dumps({k:v[k] for k in ['G_local','G_source','G_consumer','mask_attenuation','shallow_gap','masked_shape_null']}))
if __name__=='__main__':main()
