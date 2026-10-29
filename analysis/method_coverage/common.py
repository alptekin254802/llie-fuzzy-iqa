"""Human method-coverage helpers and scoped reproduction outputs."""
from __future__ import annotations
import os,sys
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
import json,hashlib,itertools,time,platform,warnings
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
threadpool_limits(1)
PUB=Path(__file__).resolve().parents[2]
RECORD=PUB/'results/studies/records'
DATASETS=['squarelol','lieq','rlie']
LABELS=dict(squarelol='SQUARE-LOL',lieq='LIEQ',rlie='RLIE')
MODELS=['fuzzy LS','linear-3f','LOL prior']
KS=[2,3,4,5,6]
SEED=20261029
REPEATS=20
B=2000
sys.path.insert(0,str(PUB))
import analysis.studies.common as c
import analysis.studies.human_objectives as h
ROOT=Path(os.environ.get('LLIE_METHOD_COVERAGE_WORK',str(c.OUT/'method_coverage'))).resolve()
check_roots=[PUB/'results',PUB/'models',PUB/'analysis']
if any(ROOT==p.resolve() or p.resolve() in ROOT.parents for p in check_roots):
    raise ValueError('Use a separate output directory, outside distributed records and sources')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    hh=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):hh.update(x)
    return hh.hexdigest()
def outpath(name):
    p=(ROOT/name).resolve();p.relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);return p
def save(name,obj):outpath(name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def csv(name,df):df.to_csv(outpath(name),index=False)
def check(value,message):
    if not bool(value):raise RuntimeError(message)
def snapshot_inputs():
    names=['analysis/studies/human_method_disjoint.py','analysis/studies/human_objectives.py','analysis/studies/common.py','analysis/studies/config.py','evaluation/frozen_scorer.py','models/nr_free.json','results/studies/records/human_scene_folds.csv','results/studies/records/human_method_splits.csv']
    result={n:sha(PUB/n) for n in names}
    for ds in DATASETS:
        n=f'results/studies/human_method_disjoint_{ds}.json';result[n]=sha(PUB/n)
        for prefix in ['feat','brisque']:
            n=f'{prefix}_{ds}_area256.csv';result['configured_work/'+n]=sha(c.OUT/n)
    return result


def scene_matrix(df):
    g=df.groupby('scene',sort=True).indices
    sizes={len(v) for v in g.values()};check(len(sizes)==1,'Unequal alternatives per scene')
    return np.stack([g[s] for s in sorted(g)])

def per_scene_srcc(pred,target,S):
    a=rankdata(np.asarray(pred)[S],axis=1,method='average')
    b=rankdata(np.asarray(target)[S],axis=1,method='average')
    a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))+1e-12
    return np.sum(a*b,axis=1)/den

def load(ds):
    check(ds in DATASETS,'Dataset outside explicit allowlist')
    x=h.load(ds)
    n,scenes={'squarelol':(2900,290),'lieq':(1000,100),'rlie':(1540,154)}[ds]
    check(len(x)==n and x.scene.nunique()==scenes and x.method.astype(str).nunique()==10,ds+': expected10 methods and fixed counts')
    check(not x.name.duplicated().any() and x.groupby('scene').size().eq(10).all(),ds+': unique complete scenes')
    check(x.groupby('scene').method.nunique().eq(10).all(),ds+': incomplete method coverage')
    check(np.isfinite(x[c.NR+['mos']].to_numpy(float)).all(),ds+': nonfinite input')
    record=pd.read_csv(RECORD/'human_scene_folds.csv',dtype=str)
    record=record[record.dataset.eq(ds)].set_index('name').loc[x.name].reset_index()
    check(np.array_equal(record.scene.to_numpy(),x.scene.to_numpy()),ds+': S14 scene identity mismatch')
    check(np.array_equal(record.method.to_numpy(),x.method.astype(str).to_numpy()),ds+': method metadata mismatch')
    foldids=record.test_fold.to_numpy(int)
    check(len(x)==len(record) and sorted(np.unique(foldids))==list(range(5)),ds+': S14 fold records')
    # Validate the stored IDs against the source function; all experiment fits
    # below use the stored record rather than regenerated scene groups.
    for f,(tr,te) in enumerate(h.scene_folds(x)):
        check(np.array_equal(te,np.flatnonzero(foldids==f)),ds+': persisted/source scene folds differ')
    return x,foldids

def arrays(x):
    signs,mfs,theta=c.lol_frozen_model()
    return h.fuzzy_design(x,signs,mfs),np.column_stack(c.oriented(x,signs)),theta

def fit_predict(x,y,D,X3,theta,trm,tem):
    Str=scene_matrix(x.iloc[trm].reset_index(drop=True))
    Ste=scene_matrix(x.iloc[tem].reset_index(drop=True))
    yc=h.demean(y[trm],Str)
    dc=h.demean(D[trm],Str)
    th=np.linalg.solve(dc.T@dc+1e-3*np.eye(27),dc.T@yc)
    lin=make_pipeline(StandardScaler(),LinearRegression()).fit(h.demean(X3[trm],Str),yc)
    pred={'LOL prior':D[tem]@theta,'fuzzy LS':D[tem]@th,'linear-3f':lin.predict(h.demean(X3[tem],Ste))}
    check(all(np.isfinite(v).all() for v in pred.values()),'Nonfinite predictions')
    return pred,Ste,dict(theta=th.tolist(),linear_coef=lin[-1].coef_.tolist(),linear_intercept=float(lin[-1].intercept_),scaler_mean=lin[0].mean_.tolist(),scaler_scale=lin[0].scale_.tolist())
