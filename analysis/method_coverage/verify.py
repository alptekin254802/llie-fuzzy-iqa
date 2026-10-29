"""Independent method_coverage audit: parameter equations, pair-count ranks, bootstrap draws."""
from .common import *
import logging
CHECKS=[]

def record(name,ok,detail=None):
    CHECKS.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',detail=detail))
    check(ok,name+': '+repr(detail))

def direct_rho(p,y):
    def ranks(a):
        # Independent average ranks, computed from pair comparisons of four values.
        return 1+(a[:,:,None]>a[:,None,:]).sum(axis=2)+.5*((a[:,:,None]==a[:,None,:]).sum(axis=2)-1)
    a=ranks(np.asarray(p));b=ranks(np.asarray(y));a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.sqrt((a*a).sum(1)*(b*b).sum(1))
    return np.divide((a*b).sum(1),den,out=np.zeros(len(a)),where=den>0)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'verification.log',encoding='utf-8')])
    log=logging.getLogger('verify');report=dict(status='RUNNING',started_utc=now())
    try:
        config=read('config.json');record('Fixed config',sha(ROOT/'config.json')==read('config_lock.json')['sha256'])
        record('S14 numerical gate',read('preflight.json')['status']=='PASS' and all(z['status']=='PASS' for z in read('preflight.json')['checks']))
        record('Scientific source inputs unchanged',snapshot_inputs()==read('input_hashes.json'))
        results=pd.read_csv(ROOT/'results.csv');summary=pd.read_csv(ROOT/'curve_summary.csv');contrasts=pd.read_csv(ROOT/'contrasts.csv')
        record('4500 complete score cases',len(results)==4500 and results.status.eq('COMPLETE').all() and results.SRCC.notna().all())
        record('Unique score keys',not results.duplicated(['dataset','model','k','repeat','fold']).any())
        record('33 prespecified contrasts',len(contrasts)==33 and contrasts.family.eq('primary').sum()==3 and contrasts.status.eq('COMPLETE').all())
        for row in summary.itertuples():
            g=results[(results.dataset==row.dataset)&(results.model==row.model)&(results.k==row.k)]
            record(f'Curve summary {row.dataset} {row.model} k{row.k}',len(g)==100 and abs(g.SRCC.mean()-row.mean_SRCC)<1e-12 and abs(g.SRCC.std(ddof=0)-row.SD)<1e-12)
        for ds in DATASETS:
            x,foldids=load(ds);D,X3,prior=arrays(x);y=x.mos.to_numpy(float);meth=x.method.astype(str).to_numpy()
            predictions=pd.read_csv(ROOT/f'predictions_{ds}.csv',dtype={'scene':str},float_precision='round_trip')
            scene_scores=pd.read_csv(ROOT/f'per_scene_{ds}.csv',dtype={'scene':str},float_precision='round_trip')
            p_groups={key:g for key,g in predictions.groupby(['repeat','fold','k'],sort=False)}
            s_groups={key:g for key,g in scene_scores.groupby(['repeat','fold','k'],sort=False)}
            states={}
            with (ROOT/f'models/{ds}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    item=json.loads(line);states[(item['repeat'],item['fold'],item['k'],item['model'])]=item
            record(ds+'1000 human model states',len(states)==1000)
            rng=np.random.default_rng(SEED);methods=np.array(config['datasets'][ds]['methods'])
            for plan in config['datasets'][ds]['splits']:
                rep=plan['repeat'];test=sorted(rng.choice(methods,4,replace=False).tolist())
                order=rng.permutation([m for m in methods if m not in test]).tolist()
                record(f'{ds} repeat{rep} seeded method selection',test==plan['test_methods'] and order==plan['addition_order'])
                for k in KS:
                    train_methods=plan['train_methods_by_k'][str(k)]
                    record(f'{ds} repeat{rep} k{k} nested/disjoint',train_methods==order[:k] and not set(train_methods)&set(test))
                for fold in range(5):
                    te=np.flatnonzero((foldids==fold)&np.isin(meth,test));xt=x.iloc[te].reset_index(drop=True);S=scene_matrix(xt)
                    test_names=x.name.iloc[te].to_numpy();prior_previous=None
                    for k in KS:
                        group=p_groups[(rep,fold,k)];scores=s_groups[(rep,fold,k)]
                        record(f'{ds} r{rep} f{fold} k{k} fixed test rows',np.array_equal(group.name.to_numpy(),test_names) and group.scene.nunique()==S.shape[0])
                        tr=np.flatnonzero((foldids!=fold)&np.isin(meth,plan['train_methods_by_k'][str(k)]))
                        record(f'{ds} r{rep} f{fold} k{k} train isolation',not set(x.scene.iloc[tr])&set(x.scene.iloc[te]) and not set(meth[tr])&set(meth[te]))
                        Str=scene_matrix(x.iloc[tr].reset_index(drop=True));yc=h.demean(y[tr],Str)
                        expected_prior=D[te]@prior
                        record(f'{ds} r{rep} f{fold} k{k} frozen prior',np.max(np.abs(group['LOL prior'].to_numpy()-expected_prior))<1e-10)
                        if prior_previous is not None:record(f'{ds} r{rep} f{fold} k{k} prior k invariance',np.array_equal(group['LOL prior'].to_numpy(),prior_previous))
                        prior_previous=group['LOL prior'].to_numpy()
                        th=np.asarray(states[(rep,fold,k,'fuzzy LS')]['theta']);dc=h.demean(D[tr],Str)
                        residual=dc.T@(dc@th-yc)+1e-3*th
                        scale=1+np.max(np.abs(dc.T@yc))
                        record(f'{ds} r{rep} f{fold} k{k} fuzzy LS normal equation',np.max(np.abs(residual))/scale<1e-9)
                        expected_fuzzy=D[te]@th
                        record(f'{ds} r{rep} f{fold} k{k} fuzzy prediction',np.max(np.abs(expected_fuzzy-group['fuzzy LS'].to_numpy()))<1e-9)
                        state=states[(rep,fold,k,'linear-3f')];coef=np.asarray(state['coef']);offset=float(state['intercept']);mu=np.asarray(state['scaler_mean']);scale=np.asarray(state['scaler_scale'])
                        centered=h.demean(X3[tr],Str);z=(centered-mu)/scale;res=z@coef+offset-yc
                        norm=1+np.max(np.abs(z.T@yc))
                        record(f'{ds} r{rep} f{fold} k{k} linear normal equation',np.max(np.abs(z.T@res))/norm<1e-9 and abs(np.mean(res))<1e-8)
                        test_features=X3[te].copy()
                        for indices in S:test_features[indices]-=test_features[indices].mean(axis=0)
                        expected_linear=((test_features-mu)/scale)@coef+offset
                        record(f'{ds} r{rep} f{fold} k{k} linear prediction',np.max(np.abs(expected_linear-group['linear-3f'].to_numpy()))<1e-9)
                        for model in MODELS:
                            p=group[model].to_numpy()[S];target=group.mos.to_numpy()[S]
                            rho=direct_rho(p,target)
                            record(f'{ds} r{rep} f{fold} k{k} {model} independent scene ranks',np.max(np.abs(rho-scores[model].to_numpy()))<1e-11)
                            row=results[(results.dataset==ds)&(results.model==model)&results.k.eq(k)&results['repeat'].eq(rep)&results.fold.eq(fold)].iloc[0]
                            record(f'{ds} r{rep} f{fold} k{k} {model} independent fold SRCC',abs(rho.mean()-row.SRCC)<1e-11)
            log.info('%s: all frozen models, fits, forward predictions, and independent ranks pass',ds)
            # The same scene must carry every repeated observation together.
            scenes=np.array(sorted(scene_scores.scene.unique()));ns=len(scenes)
            sampled=np.load(ROOT/f'bootstrap_scene_indices_{ds}.npy')
            record(ds+'bootstrap seed/whole scene count',np.array_equal(sampled,np.random.default_rng(SEED).integers(0,ns,size=(2000,ns))))
            fold=scene_scores.groupby('scene').fold.first().loc[scenes].to_numpy(int)
            effects=pd.read_csv(ROOT/'contrast_scene_effects.csv',dtype={'scene':str},float_precision='round_trip')
            effects=effects[effects.dataset.eq(ds)]
            raw=pd.read_csv(ROOT/'bootstrap_differences.csv',float_precision='round_trip');raw=raw[raw.dataset.eq(ds)].reset_index(drop=True)
            for row in contrasts[contrasts.dataset.eq(ds)].itertuples():
                a=scene_scores[scene_scores.k.eq(row.k)].pivot(index='scene',columns='repeat',values=row.model).loc[scenes,list(range(20))].to_numpy()
                b=scene_scores[scene_scores.k.eq(row.reference_k)].pivot(index='scene',columns='repeat',values=row.reference_model).loc[scenes,list(range(20))].to_numpy()
                delta=(a-b).mean(axis=1)
                archived=effects[effects.contrast.eq(row.contrast)].set_index('scene').loc[scenes]
                record(ds+'paired20-repeat scene effect '+row.contrast,np.max(np.abs(delta-archived.mean_difference_over20_repeats.to_numpy()))<1e-12)
                point=float(np.mean([np.mean(delta[fold==f]) for f in range(5)]))
                record(ds+'point estimand '+row.contrast,abs(point-row.difference)<1e-12)
                for draw in [0,7,1999]:
                    ids=sampled[draw];value=float(np.mean([np.mean(delta[ids[fold[ids]==f]]) for f in range(5)]))
                    record(ds+f'bootstrap direct draw{draw} '+row.contrast,abs(value-raw[row.contrast].iloc[draw])<1e-12)
                ci=np.quantile(raw[row.contrast],[.025,.975])
                record(ds+'CI endpoint '+row.contrast,np.max(np.abs(ci-np.array([row.ci_low,row.ci_high])))<1e-12)
        record('Scientific inputs still unchanged',snapshot_inputs()==read('input_hashes.json'))
        report.update(status='PASS',finished_utc=now(),checks=len(CHECKS),passed=sum(x['status']=='PASS' for x in CHECKS),failed=0,failed_runs=int(results.status.ne('COMPLETE').sum()))
        save('verification.json',report);csv('verification_checks.csv',pd.DataFrame(CHECKS))
        status=read('run_status.json');status.update(status='COMPLETE',verification=report['status'],verified_utc=now());save('run_status.json',status)
        log.info('%s: %d checks; zero failed model runs',report['status'],report['passed'])
    except Exception as e:
        report.update(status='FAIL',error=str(e),finished_utc=now(),checks=len(CHECKS));save('verification.json',report);csv('verification_checks.csv',pd.DataFrame(CHECKS));raise
if __name__=='__main__':main()
