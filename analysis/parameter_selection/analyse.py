"""Recompute grouped means, bootstrap intervals and paired tests without images."""
import argparse,itertools,json
import numpy as np
import pandas as pd
from .common import DATA,read,csv,output_dir

def paired_p(a,seed):
    a=np.asarray(a);n=len(a);obs=abs(a.mean());tol=1e-15
    if n<=10:
        signs=np.array(list(itertools.product([-1.,1.],repeat=n)))
        return float((abs(signs@a/n)>=obs-tol).mean()),'exact_sign_flip'
    rng=np.random.default_rng(seed);count=0;N=19999
    for start in range(0,N,500):
        signs=rng.choice([-1.,1.],size=(min(500,N-start),n))
        count+=int(np.sum(abs(signs@a/n)>=obs-tol))
    return (count+1)/(N+1),'monte_carlo_sign_flip'

def calculate(part,group_file=None):
    groups=read(group_file or DATA/part/'group_scores.csv')
    recorded=read(DATA/part/'metrics.csv')
    estimates=[];contrasts=[];primary=[]
    keys=['dataset','family'] if part=='screen' else ['dataset']
    for di,(key,z) in enumerate(groups.groupby(keys,sort=True)):
        ds=key[0];fam=key[1] if part=='screen' else 'clahe'
        reference_groups=sorted(z.reference_group.unique());n=len(reference_groups)
        boot=np.random.default_rng(20261030+di).integers(n,size=(2000,n))
        metadata={'dataset':ds,**({'family':fam} if part=='screen' else {})}
        sample=recorded[recorded.dataset==ds]
        if part=='screen':sample=sample[sample.family==fam]
        scenes=int(sample.scenes.iloc[0])
        for metric,g in z.groupby('metric',sort=True):
            w=g.set_index('reference_group').reindex(reference_groups).drop(columns=['dataset','family','metric'])
            w=w.loc[:,w.notna().all()]
            for selector in sorted(w):
                a=w[selector].to_numpy(float);lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
                estimates.append(dict(**metadata,metric=metric,selector=selector,mean=a.mean(),ci_low=lo,ci_high=hi,groups=n,scenes=scenes))
            for model in (['NR-free'] if part=='screen' else ['NR-mono','NR-free']):
                for comparator in sorted(w):
                    if comparator==model:continue
                    a=(w[model]-w[comparator]).to_numpy(float)
                    lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
                    row=dict(**metadata,metric=metric,**({'model':model} if part=='confirmation' else {}),comparator=comparator,contrast=model+' minus '+comparator,mean=a.mean(),ci_low=lo,ci_high=hi,groups=n)
                    is_primary=metric=='lpips' and ((part=='screen' and comparator=='training_fixed') or (part=='confirmation' and model=='NR-mono' and comparator in ['training_fixed','default','Linear3','Linear4','RF3','RF4']))
                    if is_primary:
                        pv,method=paired_p(a,20261030+di)
                        row.update(p_raw=pv if part=='screen' or n>=20 else 1.,p_method=method)
                        if part=='screen':row['primary_domain']='development' if ds=='LOL' else 'external'
                        else:row['n_below_confirmatory_minimum']=bool(n<20)
                        primary.append(row)
                    contrasts.append(row)
    families=([r for r in primary if r['primary_domain']=='development'],[r for r in primary if r['primary_domain']=='external']) if part=='screen' else (primary,)
    for family in families:
        prior=0
        for i,r in enumerate(sorted(family,key=lambda r:r['p_raw'])):
            prior=min(1,max(prior,r['p_raw']*(len(family)-i)));r['p_holm']=prior
            r['strong_bounded_candidate' if part=='screen' else 'positive_lpips_comparison']=bool(r['mean']<=-.005 and prior<.05)
    return {'metrics':pd.DataFrame(estimates),'contrasts':pd.DataFrame(contrasts),'primary_tests':pd.DataFrame(primary)}

def compare(part,tables):
    counts={}
    for name,actual in tables.items():
        expected=read(DATA/part/(name+'.csv'))
        keys=['dataset','metric']+(['family'] if part=='screen' else [])+(['selector'] if name=='metrics' else ['comparator']+(['model'] if part=='confirmation' else []))
        expected=expected.set_index(keys).sort_index();actual=actual.set_index(keys).sort_index()
        if not expected.index.equals(actual.index):raise AssertionError((part,name,'row keys differ'))
        for col in ['mean','ci_low','ci_high','p_raw','p_holm']:
            if col in expected:
                if not np.allclose(actual[col],expected[col],atol=1e-12,rtol=0,equal_nan=True):raise AssertionError((part,name,col,float(np.nanmax(abs(actual[col]-expected[col])))))
        counts[name]=len(actual)
    return counts

def secondary():
    groups=read(DATA/'screen/group_scores.csv');rows=[]
    for di,((ds,fam),z) in enumerate(groups.groupby(['dataset','family'],sort=True)):
        ids=sorted(z.reference_group.unique());n=len(ids)
        boot=np.random.default_rng(20261030+di).integers(n,size=(2000,n))
        for metric,g in z.groupby('metric',sort=True):
            w=g.set_index('reference_group').reindex(ids)
            for model in ['NR-mono','BrightnessCorrection_lam01']:
                for comparator in ['training_fixed','default','entropy','brightness_tuned','Linear3','Linear4','test_best_fixed_oracle']:
                    if w[comparator].isna().any():continue
                    a=(w[model]-w[comparator]).to_numpy(float);lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
                    rows.append(dict(dataset=ds,family=fam,metric=metric,model=model,comparator=comparator,contrast=model+' minus '+comparator,mean=a.mean(),ci_low=lo,ci_high=hi,groups=n,analysis='post_hoc_secondary'))
    actual=pd.DataFrame(rows);keys=['dataset','family','metric','model','comparator']
    x=actual.set_index(keys).sort_index();y=read(DATA/'screen/secondary_contrasts.csv').set_index(keys).sort_index()
    assert x.index.equals(y.index)
    assert np.allclose(x[['mean','ci_low','ci_high']],y[['mean','ci_low','ci_high']],atol=1e-12,rtol=0)
    return actual

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['screen','confirmation','both'],default='both');p.add_argument('--output');a=p.parse_args()
    for part in (['screen','confirmation'] if a.part=='both' else [a.part]):
        tables=calculate(part);print(part,compare(part,tables))
        if part=='screen':tables['secondary_contrasts']=secondary();print('secondary contrasts',len(tables['secondary_contrasts']))
        if a.output:
            d=output_dir(a.output)
            for name,table in tables.items():csv(d/part/(name+'.csv'),table)
if __name__=='__main__':main()
