"""Whole-scene paired bootstrap across the 20-repeat method-repeat realizations."""
from .common import *
import logging

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'analysis.log',encoding='utf-8')])
    log=logging.getLogger('analysis')
    check(read('run_status.json')['status'] in ['MODELS_DONE','ANALYZED'],'Fit stage incomplete')
    check(sha(ROOT/'config.json')==read('config_lock.json')['sha256'],'Config changed')
    results=pd.read_csv(ROOT/'results.csv')
    summaries=[]
    for (ds,model,k),g in results.groupby(['dataset','model','k'],sort=False):
        good=g.status.eq('COMPLETE')
        vals=g.loc[good,'SRCC'].to_numpy(float)
        summaries.append(dict(dataset=ds,model=model,k=int(k),mean_SRCC=float(np.mean(vals)) if len(vals) else np.nan,SD=float(np.std(vals,ddof=0)) if len(vals) else np.nan,n_repeats=20,n_folds=5,n_cases=len(g),n_complete=int(good.sum()),n_failed=int((~good).sum()),status='COMPLETE' if good.all() else 'INCOMPLETE'))
    summary=pd.DataFrame(summaries);csv('curve_summary.csv',summary)
    contrasts=[];draw_parts=[];effect_parts=[]
    for ds in DATASETS:
        per=pd.read_csv(ROOT/f'per_scene_{ds}.csv',dtype={'scene':str},float_precision='round_trip')
        check(not per.duplicated(['repeat','k','scene']).any(),'Duplicate scene score')
        scenes=np.array(sorted(per.scene.unique()));n=len(scenes)
        fold=per.groupby('scene',sort=True).fold.first().loc[scenes].to_numpy(int)
        mean_scores={}
        for k in KS:
            mean_scores[k]={}
            part=per[per.k.eq(k)]
            for model in MODELS:
                matrix=part.pivot(index='scene',columns='repeat',values=model).loc[scenes,list(range(20))].to_numpy(float)
                mean_scores[k][model]=np.mean(matrix,axis=1)
        descriptors=[dict(family='primary',model='fuzzy LS',k=6,reference_model='fuzzy LS',reference_k=2,contrast='fuzzy_LS_k6_minus_k2')]
        for k in KS:
            for name in ['fuzzy LS','linear-3f']:
                descriptors.append(dict(family='secondary',model=name,k=k,reference_model='LOL prior',reference_k=k,contrast=name.replace(' ','_')+f'_k{k}_minus_LOL_prior'))
        effects=np.column_stack([mean_scores[d['k']][d['model']]-mean_scores[d['reference_k']][d['reference_model']] for d in descriptors])
        rng=np.random.default_rng(SEED);indices=rng.integers(0,n,size=(B,n))
        np.save(outpath(f'bootstrap_scene_indices_{ds}.npy'),indices)
        counts=np.zeros((B,n),np.int32)
        np.add.at(counts,(np.repeat(np.arange(B),n),indices.ravel()),1)
        weights=np.zeros((B,n),float);point_weights=np.zeros(n)
        for f in range(5):
            mask=fold==f;den=counts[:,mask].sum(axis=1)
            check(np.all(den>0),'Bootstrap draw omitted a fixed fold')
            weights[:,mask]=counts[:,mask]/den[:,None]/5
            point_weights[mask]=1/(5*mask.sum())
        draws=weights@effects;point=point_weights@effects
        raw=pd.DataFrame(draws,columns=[d['contrast'] for d in descriptors]);raw.insert(0,'draw',np.arange(B));raw.insert(0,'dataset',ds);draw_parts.append(raw)
        for j,d in enumerate(descriptors):
            finite=np.isfinite(draws[:,j]).all() and np.isfinite(point[j])
            lo,hi=np.quantile(draws[:,j],[.025,.975]) if finite else (np.nan,np.nan)
            a=summary[(summary.dataset==ds)&(summary.model==d['model'])&summary.k.eq(d['k'])].iloc[0]
            b=summary[(summary.dataset==ds)&(summary.model==d['reference_model'])&summary.k.eq(d['reference_k'])].iloc[0]
            if finite:check(abs(point[j]-(a.mean_SRCC-b.mean_SRCC))<1e-12,'Bootstrap point differs from equal repeat-fold mean')
            contrasts.append(dict(dataset=ds,**d,difference=float(point[j]),ci_low=float(lo),ci_high=float(hi),model_mean=a.mean_SRCC,reference_mean=b.mean_SRCC,repeats=B,seed=SEED,scene_count=n,method_repeats=20,folds=5,interval='percentile95%, marginal unadjusted',unit='whole scene; all20 method repeats paired; fixed folds retained',status='COMPLETE' if finite else 'INCOMPLETE_FAILED_CASES'))
            effect_parts.append(pd.DataFrame(dict(dataset=ds,contrast=d['contrast'],scene=scenes,fold=fold,mean_difference_over20_repeats=effects[:,j],point_weight=point_weights)))
        log.info('%s:11 paired contrasts,2000 whole-scene draws complete',ds)
    csv('contrasts.csv',pd.DataFrame(contrasts));csv('bootstrap_differences.csv',pd.concat(draw_parts,ignore_index=True));csv('contrast_scene_effects.csv',pd.concat(effect_parts,ignore_index=True))
    plot(summary)
    report(summary,pd.DataFrame(contrasts),results)
    status=read('run_status.json');status.update(status='ANALYZED',analysis_finished_utc=now(),contrasts=len(contrasts));save('run_status.json',status)

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(12.0,4.0),sharex=True,sharey=True)
    colors={'fuzzy LS':'#1768AC','linear-3f':'#D86B16','LOL prior':'#555555'}
    labels={'fuzzy LS':'Fuzzy LS','linear-3f':'Linear-3f','LOL prior':'LOL prior'}
    for ax,ds in zip(axes,DATASETS):
        for model in MODELS:
            g=summary[(summary.dataset==ds)&(summary.model==model)].sort_values('k')
            x=g.k.to_numpy();mean=g.mean_SRCC.to_numpy();sd=g.SD.to_numpy()
            ax.plot(x,mean,'o-' if model!='LOL prior' else 's--',color=colors[model],lw=1.8,ms=4,label=labels[model])
            ax.fill_between(x,mean-sd,mean+sd,color=colors[model],alpha=.10,linewidth=0)
        ax.set_title(LABELS[ds]);ax.set_xticks(KS);ax.set_ylim(-1,1);ax.grid(axis='y',alpha=.22);ax.axhline(0,color='black',lw=.6,alpha=.4)
        ax.set_xlabel('Number of training methods (k)')
    axes[0].set_ylabel('Within-scene SRCC (mean +/- SD)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.075),frameon=False)
    fig.text(.5,.015,'20 repeats x 5 fixed scene folds; bands show SD over 100 fold scores, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.18,1,1));fig.savefig(outpath('coverage_curve.png'),dpi=220,facecolor='white');fig.savefig(outpath('coverage_curve.svg'),facecolor='white');plt.close(fig)

def report(summary,contrasts,results):
    lines=['# Training-method coverage','',
        'The experiment retains the S14 256x256 area-interpolated features, five stored scene folds, frozen LOL orientations and memberships, ridge=0.001 fuzzy least squares, and scene-centred linear regression. The preparation check reproduces the three S14 models on all three datasets to three decimals.',
        'Each of 20 repeats (seed 20261029) holds four test methods fixed across nested training sets of two to six methods. Training image counts increase with method counts. The LOL prior uses no human-score fitting.',
        'Values below are means and population SDs over 20 method repeats x 5 scene folds.','',
        '| Dataset | Model | k=2 | k=3 | k=4 | k=5 | k=6 |','|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for model in MODELS:
            g=summary[(summary.dataset==ds)&(summary.model==model)].set_index('k')
            cells=[f'{g.loc[k,"mean_SRCC"]:.3f} +/- {g.loc[k,"SD"]:.3f}' for k in KS]
            lines.append('| '+LABELS[ds]+' | '+model+' | '+' | '.join(cells)+' |')
    lines+=['','| Fuzzy LS: k=6 minus k=2 | Difference | 95% CI |','|---|---:|---:|']
    for ds in DATASETS:
        r=contrasts[(contrasts.dataset==ds)&contrasts.family.eq('primary')].iloc[0]
        lines.append(f'| {LABELS[ds]} | {r.difference:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
    lines+=['',
        'Intervals use 2,000 paired whole-scene bootstrap draws (seed 20261029). All 20 repeated observations of a scene move together. Fixed scene folds retain equal weight. Fits and method splits stay fixed; intervals are marginal and unadjusted.',
        'Differences are the six-method fit minus the two-method fit; positive values favour the six-method fit. Secondary contrasts compare each human-calibrated fit with the LOL prior at the same coverage level.',
        'Method diversity and training image count change together, so this test does not isolate their effects. The test compares four alternatives per scene; its LOL-prior values differ from the five-method S14 evaluation.',
        f'Completed score conditions: {int(results.status.eq("COMPLETE").sum())}/{len(results)}; failed conditions: {int(results.status.ne("COMPLETE").sum())}.',
        'See README.md for reproduction commands and contrasts.csv for all paired intervals.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\r\n')
if __name__=='__main__':main()
