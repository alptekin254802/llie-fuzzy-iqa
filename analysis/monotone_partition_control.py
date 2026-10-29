"""Compare global and training-fold partition/sign estimation for NR-mono."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from analysis.monotone_fuzzy import fit,design
from analysis.reference_free_fuzzy import mfset,group_kfold,NR_FEATURES
def main():
    threadpool_limits(1);root=Path(__file__).resolve().parents[1];res=root/'results'
    d=pd.read_csv(res/'nr_features.csv').merge(pd.read_csv(res/'perceptual_metrics.csv')[['filename','method','lpips']],on=['filename','method'],validate='one_to_one')
    y=-d.lpips.to_numpy();X=d[NR_FEATURES].to_numpy();out={}
    folds=list(group_kfold(d.filename,5,42))
    for setting in ('global','train_fold'):
        models=[];p=np.zeros(len(d));values=[]
        for i,(tr,te) in enumerate(folds):
            rows=np.arange(len(d)) if setting=='global' else tr
            signs=np.array([-1. if spearmanr(X[rows,n],y[rows]).statistic<0 else 1. for n in range(3)])
            A=X*signs;mf=[mfset(A[rows,n]) for n in range(3)]
            theta=fit(A[tr],y[tr],mf);p[te]=design(A[te],mf)@theta
            values.append(float(spearmanr(p[te],y[te]).statistic));models.append({'signs':signs.tolist(),'mfs':mf,'theta':theta.tolist()})
        out[setting]={'folds':values,'mean':float(np.mean(values)),'std':float(np.std(values)),'models':models}
        d[['filename','method']].assign(prediction=p).to_csv(res/f'monotone_{setting}_control.csv',index=False)
    out['train_minus_global']=out['train_fold']['mean']-out['global']['mean']
    (res/'monotone_partition_control.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:{n:v for n,v in r.items() if n!='models'} if isinstance(r,dict) else r for k,r in out.items()})
if __name__=='__main__':main()
