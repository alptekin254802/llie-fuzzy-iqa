"""Exact CLAHE search: fixed pixels/features/model weights, fewer conversions.

API: Engine(models=None).search(raw, selector) -> (parameter, float32 RGB).
inspect(raw, selector='NR-mono') exposes all nine candidates/features/scores.
Instances are intended for sequential single-thread calls; no image-result cache.
"""
from pathlib import Path
import sys,itertools,importlib
import numpy as np
import cv2

PARAMS=np.array([-1.,.5,1.,1.5,2.,3.,4.,6.,8.],dtype=np.float64)
FEATURE_NAMES=('entropy','contrast','sharpness')
SELECTORS=('NR-free','NR-mono','Linear3')

def choose(scores):
    scores=np.asarray(scores,float)
    mask=scores>=scores.max()-1e-9
    distance=np.round(abs(PARAMS-2.),12)
    mask &= distance==np.where(mask,distance,np.inf).min()
    return int(np.where(mask,PARAMS,np.inf).argmin())

def rgb8_to_float(rgb):
    # Division is intentional: reciprocal multiplication is not byte-identical.
    return np.ascontiguousarray(rgb.astype(np.float32)/255.,dtype=np.float32)

def features_rgb8(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hist=cv2.calcHist([np.ascontiguousarray(gray)],[0],None,[256],[0,256]).ravel()
    p=hist/hist.sum();p=p[p>0]
    entropy=float(-(p*np.log2(p)).sum())
    g=gray.astype(float)
    contrast=float(g.std()/255)
    sharpness=float(cv2.Laplacian(g,cv2.CV_64F).var()/255**2)
    return entropy,contrast,sharpness

class Engine:
    def __init__(self,models=None):
        from analysis.parameter_selection.scorers import load_models, firing
        from analysis.parameter_selection.candidates import exposed
        if models is None: models = load_models('final')
        self.models={name:models[name] for name in SELECTORS}
        self.exposed=exposed
        self.firing=firing
        self.m0=self.models['NR-free']
        self.mono=self.models['NR-mono'].released
        p=self.models['Linear3'].parameters
        assert tuple(p['feature_order'])==FEATURE_NAMES
        self.linear_mean=np.asarray(p['mean'],float);self.linear_scale=np.asarray(p['scale'],float)
        self.linear_coefficient=np.asarray(p['coefficient'],float);self.linear_intercept=float(p['intercept'])
        self.clahe={float(p):cv2.createCLAHE(clipLimit=float(p),tileGridSize=(8,8)) for p in PARAMS[1:]}
        # Exhaustive channel-value proof for omitting RGB8->float32->RGB8 in features.
        v=np.arange(256,dtype=np.uint8)
        assert np.array_equal(np.clip((v.astype(np.float32)/255.)*255,0,255).astype(np.uint8),v)

    def _prepare(self,raw):
        if not isinstance(raw,np.ndarray) or raw.shape!=(256,256,3) or raw.dtype!=np.float32:
            raise ValueError('Expected RGB256 float32 input')
        if not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
            raise ValueError('Expected finite RGB values in [0,1]')
        # The exact original roundtrip remains inside the timed boundary.
        low=self.exposed(raw,1.)
        pixels=np.clip(low*255.,0.,255.).astype(np.uint8)
        return low,pixels

    def _candidates8(self,low,pixels):
        lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB);light,a,b=cv2.split(lab)
        result=[pixels]
        for p in PARAMS[1:]:
            enhanced=self.clahe[float(p)].apply(light)
            result.append(cv2.cvtColor(cv2.merge([enhanced,a,b]),cv2.COLOR_LAB2RGB))
        return result

    def _features(self,candidates):
        return np.array([features_rgb8(rgb) for rgb in candidates],dtype=np.float64)

    def score(self,features,selector):
        x=np.asarray(features,dtype=np.float64)
        if x.shape!=(9,3):raise ValueError('Expected nine rows of the three original features')
        if selector=='NR-free':
            fire,den=self.firing(x,self.m0.signs,self.m0.mfs)
            return (fire@self.m0.theta)/den
        if selector=='NR-mono':return self.mono.predict(x)
        if selector=='Linear3':return ((x-self.linear_mean)/self.linear_scale)@self.linear_coefficient+self.linear_intercept
        raise ValueError(selector)

    def search(self,raw,selector='NR-mono'):
        low,pixels=self._prepare(raw)
        if selector in ('training_fixed','default'):
            p=4. if selector=='training_fixed' else 2.
            lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB);light,a,b=cv2.split(lab)
            enhanced=self.clahe[p].apply(light)
            rgb=cv2.cvtColor(cv2.merge([enhanced,a,b]),cv2.COLOR_LAB2RGB)
            return p,rgb8_to_float(rgb)
        candidates=self._candidates8(low,pixels);x=self._features(candidates)
        index=choose(self.score(x,selector))
        output=low.copy() if index==0 else rgb8_to_float(candidates[index])
        return float(PARAMS[index]),output

    def inspect(self,raw,selector='NR-mono'):
        low,pixels=self._prepare(raw);rgb=self._candidates8(low,pixels);x=self._features(rgb)
        floats=[low.copy()]+[rgb8_to_float(v) for v in rgb[1:]]
        scores={name:self.score(x,name) for name in SELECTORS}
        selected={name:dict(index=choose(v),param=float(PARAMS[choose(v)]),output=floats[choose(v)]) for name,v in scores.items()}
        assert selector in SELECTORS
        return dict(params=PARAMS.copy(),candidate_uint8=rgb,candidate_float32=floats,
                    feature_names=FEATURE_NAMES,features=x,scores_by_selector=scores,
                    selected_by_selector=selected,scores=scores[selector],selected_index=selected[selector]['index'],
                    selected_param=selected[selector]['param'],selected_float32=selected[selector]['output'])
