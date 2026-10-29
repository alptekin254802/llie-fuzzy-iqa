"""Freeze method coverage design and reproduce the three S14 models."""
from .common import *
import logging,importlib.metadata as md

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'prepare.log',encoding='utf-8')])
    log=logging.getLogger('prepare')
    report=dict(status='RUNNING',started_utc=now(),checks=[])
    gate_rows=[]
    save('preflight.json',report)
    try:
        check(not (ROOT/'config.json').exists(),'Do not replace fixed method_coverage config')
        save('input_hashes.json',snapshot_inputs())
        old_methods=pd.read_csv(RECORD/'human_method_splits.csv',dtype={'method':str})
        configs={};gate_rows=[];recorded_folds=[];split_rows=[]
        for ds in DATASETS:
            x,foldids=load(ds);D,X3,prior=arrays(x);y=x.mos.to_numpy(float);meth=x.method.astype(str).to_numpy()
            methods=np.array(sorted(set(meth)))
            old=old_methods[old_methods.dataset.eq(ds)]
            values={m:[] for m in MODELS}
            for rep in range(10):
                selected=old[old['repeat'].eq(rep)]
                train=set(selected.loc[selected.role.eq('train'),'method']);test=set(selected.loc[selected.role.eq('test'),'method'])
                check(len(train)==5 and len(test)==5 and not train&test,'S14 archived five-vs-five split')
                for f in range(5):
                    tr=np.flatnonzero((foldids!=f)&np.isin(meth,list(train)))
                    te=np.flatnonzero((foldids==f)&np.isin(meth,list(test)))
                    preds,S,_=fit_predict(x,y,D,X3,prior,tr,te)
                    for name,p in preds.items():values[name].append(h.within_srcc(p,y[te],S))
            old_result=json.loads((PUB/f'results/studies/human_method_disjoint_{ds}.json').read_text())
            ref=next(v for k,v in old_result.items() if k.startswith('(b)'))
            for name,v in values.items():
                key='fuzzy LOL prior (no labels)' if name=='LOL prior' else name
                actual=[float(np.mean(v)),float(np.std(v,ddof=0))];expected=ref[key]
                good=all(f'{a:.3f}'==f'{e:.3f}' for a,e in zip(actual,expected))
                row=dict(dataset=ds,model=name,mean=actual[0],sd=actual[1],expected_mean=expected[0],expected_sd=expected[1],status='PASS' if good else 'FAIL');gate_rows.append(row)
                check(good,'S14 reproduction mismatch: '+repr(row))
            rng=np.random.default_rng(SEED);splits=[]
            for rep in range(REPEATS):
                test=sorted(rng.choice(methods,4,replace=False).tolist())
                remaining=[m for m in methods if m not in test]
                order=rng.permutation(remaining).tolist()
                plans={str(k):order[:k] for k in KS}
                splits.append(dict(repeat=rep,test_methods=test,addition_order=order,train_methods_by_k=plans))
                for method in methods:split_rows.append(dict(dataset=ds,repeat=rep,method=method,role='test' if method in test else 'training_pool',addition_position=0 if method in test else order.index(method)+1))
            configs[ds]=dict(images=len(x),scenes=x.scene.nunique(),methods=methods.tolist(),method_count=10,scene_fold_counts={str(f):int(x.loc[foldids==f,'scene'].nunique()) for f in range(5)},splits=splits)
            recorded_folds.append(x[['name','scene']].assign(dataset=ds,test_fold=foldids))
            log.info('%s: S14 reproduced;10 methods; stored scene folds verified',ds)
        config=dict(task='human_method_coverage',created_utc=now(),seed=SEED,repeats=20,scene_folds=5,k_values=KS,test_methods_per_repeat=4,training_pool_methods=6,nesting='k uses first k of the single random six-method addition order; same four test methods for all k and folds in that repeat',randomization='np.random.default_rng(20261029), reset per dataset; choose4 test without replacement, then permute remaining6',datasets=configs,
            source=dict(S14_code='analysis/studies/human_method_disjoint.py',helper='analysis/studies/human_objectives.py',preprocessing='analysis/studies/common.py',scene_fold_source='results/studies/records/human_scene_folds.csv',old_method_split_source='results/studies/records/human_method_splits.csv',reference='supplement section sup:recalibration, table_human_methods.tex',hashes='input_hashes.json'),
            models=dict(fuzzy_LS='27 rules; frozen LOL orientation/MFs; minimum firing and sum+1e-9 normalization; scene-center design and MOS across k training methods; ridge1e-3 to zero; raw test design times coefficients',linear_3f='three frozen-LOL-oriented features; scene-center training features/targets across k methods; StandardScaler+LinearRegression; test features centered within the four test alternatives; no test-label use',LOL_prior='same fixed LOL model and test rows for all k; no human-label calibration'),
            preprocessing='existing256x256 INTER_AREA feature caches, same S14 loader/BRISQUE inner join/order and scene conversion; no image loading or feature extraction',
            metric=dict(name='mean within-scene SRCC over four held-out methods',ties='S14 current average ranks; denominator sqrt(norm2p*norm2y)+1e-12; constant prediction/target scores produce zero',curve='mean and population SD (ddof0) over20 repeats x5 folds =100 values per dataset/model/k'),
            bootstrap=dict(repeats=2000,seed=SEED,unit='whole original scene; all20 method repeats/models/k kept paired',method='average per-scene differences over20 repeats; resample original scene IDs with replacement; preserve source fold labels and take equal mean of5 resampled fold means',estimand='equal mean over20 repeats x5 fixed folds, matching the coverage curve',models_fixed=True,refit=False,method_repeat_resampling=False,CI='percentile2.5,97.5, marginal unadjusted',draws_reused_across_all_contrasts_in_dataset=True),
            prespecified_contrasts=dict(primary='fuzzy LS k6 minus k2, one per dataset',secondary='for each k, fuzzy LS minus LOL prior and linear-3f minus LOL prior',counts=dict(primary=3,secondary=30)),
            environment=dict(executable=Path(sys.executable).name,python=sys.version,platform=platform.platform(),threads=1,packages={name:md.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib']}),output_scope='separate reproduction directory')
        save('config.json',config);save('config_lock.json',dict(sha256=sha(ROOT/'config.json'),created_utc=now()))
        csv('method_splits.csv',pd.DataFrame(split_rows));csv('scene_folds.csv',pd.concat(recorded_folds,ignore_index=True));csv('s14_reproduction.csv',pd.DataFrame(gate_rows))
        report.update(status='PASS',finished_utc=now(),checks=gate_rows)
        save('preflight.json',report)
        check(snapshot_inputs()==read('input_hashes.json'),'Inputs changed during preparation')
    except Exception as e:
        report.update(status='STOPPED',error=str(e),finished_utc=now());save('preflight.json',report)
        if gate_rows:csv('s14_reproduction.csv',pd.DataFrame(gate_rows))
        (ROOT/'REPORT.md').write_text('Preparation stopped: '+str(e)+'\n',encoding='utf-8')
        raise
if __name__=='__main__':main()
