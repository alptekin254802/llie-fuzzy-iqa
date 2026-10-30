"""Frozen model inference and fixed classical-only random-forest controls."""
from functools import lru_cache
import itertools,json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from evaluation.frozen_scorer import FrozenScorer
from analysis.monotone_fuzzy import design
from .common import DATA,read

FEATURES=('entropy','contrast','sharpness','brightness')
NAMES=('NR-free','NR-mono','BrightnessCorrection_lam01','Linear3','Linear4')

def trapmf(x,a,b,c,d):
    left=np.where(b>a,(x-a)/np.maximum(b-a,1e-12),1.)
    right=np.where(d>c,(d-x)/np.maximum(d-c,1e-12),1.)
    return np.clip(np.minimum(np.minimum(left,1.),right),0.,1.)

def firing(x,signs,mfs):
    x=np.asarray(x,float)*signs
    mu=[np.stack([trapmf(x[:,j],*mf) for mf in mfs[j]],axis=1) for j in range(x.shape[1])]
    rules=list(itertools.product(range(3),repeat=x.shape[1]))
    fire=np.stack([np.minimum.reduce([mu[j][:,r[j]] for j in range(len(mu))]) for r in rules],axis=1)
    return fire,fire.sum(axis=1)+1e-9

class ArrayScorer:
    def __init__(self,p):
        self.parameters=p;self.features=tuple(p['feature_order']);self.kind=p['kind']
        for k in ['signs','mfs','theta','mean','scale','coefficient']:
            if k in p:setattr(self,k,np.asarray(p[k],float))
        if self.kind=='released_mono':self.released=FrozenScorer(p['released_parameters'])
        if self.kind=='brightness_correction':self.prior=ArrayScorer(p['prior'])
    def predict(self,df):
        x=df[list(self.features)].to_numpy(float)
        if self.kind=='linear':return ((x-self.mean)/self.scale)@self.coefficient+self.parameters['intercept']
        if self.kind=='mono_product':return design(x*self.signs,self.mfs)@self.theta
        if self.kind=='released_mono':return self.released.predict(x)
        if self.kind=='brightness_correction':
            eye=np.eye(5);knots=np.linspace(0,1,5)
            b=np.column_stack([np.interp(df.brightness,knots,eye[j]) for j in range(5)])
            return np.clip(self.prior.predict(df)+b@self.theta,0,100)
        f,den=firing(x,self.signs,self.mfs)
        return (f@self.theta)/den

@lru_cache(maxsize=6)
def load_models(scope='final'):
    registry=json.loads((DATA/'models.json').read_text())
    return {n:ArrayScorer(json.loads((DATA/registry['scopes'][scope][n]).read_text())) for n in NAMES}

def scope_for_scene(scene):return json.loads((DATA/'models.json').read_text())['scene_to_scope'][str(scene)]

@lru_cache(maxsize=1)
def forests():
    data=read(DATA/'classical_training.csv')
    settings=json.loads((DATA/'rf_settings.json').read_text());result={}
    for n in [3,4]:
        m=RandomForestRegressor(**settings);m.fit(data[list(FEATURES[:n])],-data.lpips)
        result['RF'+str(n)]=m
    return result

def score_models(df,scope='final',with_forests=False):
    models=load_models(scope)
    out=pd.DataFrame({n:m.predict(df) for n,m in models.items()},index=df.index)
    if with_forests:
        if scope!='final':raise ValueError('The RF controls are final LOL fits for external confirmation only.')
        for n,m in forests().items():out[n]=m.predict(df[list(FEATURES[:int(n[-1])])])
    if not np.isfinite(out).all().all():raise ValueError('Non-finite model prediction')
    return out
