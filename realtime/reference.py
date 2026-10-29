"""Independent floating-point reference for the camera selftest.

Copies the synthetic study operations without importing the optimized kernel.
This comparison path is excluded from the measured processing time."""
import itertools
import numpy as np
import pandas as pd
import cv2
NR=['entropy','contrast','sharpness']
COMBOS=list(itertools.product(range(3),range(3),range(3)))
def trapmf(x,a,b,c,d):
    x=np.asarray(x,float)
    left=np.where(b>a,(x-a)/np.maximum(b-a,1e-12),1.0)
    right=np.where(d>c,(d-x)/np.maximum(d-c,1e-12),1.0)
    return np.clip(np.minimum(np.minimum(left,1.0),right),0,1)
def oriented(X,signs,cols=NR):
    return [-X[c].to_numpy(float) if signs[c] else X[c].to_numpy(float) for c in cols]
def firing(arrs,mfs):
    Ms=[np.stack([trapmf(a,*mfs[n][l]) for l in range(3)],axis=1) for n,a in enumerate(arrs)]
    fire=np.stack([np.minimum(np.minimum(Ms[0][:,i],Ms[1][:,j]),Ms[2][:,k]) for i,j,k in COMBOS],axis=1)
    return fire,fire.sum(axis=1)+1e-09
def predict(model,df,product=False):
    if product:
        arr=oriented(df,model['signs']);mfs=model['mfs']
        ms=[np.stack([trapmf(arr[n],*mfs[n][j]) for j in range(3)],1) for n in range(3)]
        f=np.stack([ms[0][:,i]*ms[1][:,j]*ms[2][:,k] for i,j,k in COMBOS],1)
        return (f/(f.sum(1,keepdims=True)+1e-12))@np.array(model['theta'])
    fire,fsum=firing(oriented(df,model['signs']),model['mfs'])
    return fire@np.array(model['theta'])/fsum
def gamma_float(inp,g):
    gamma=1/g
    img_float=inp.astype(np.float32)
    corrected=np.power(img_float,1.0/gamma)
    restored=np.clip(corrected,0.,1.).astype(np.float32)
    return np.clip(restored,0,1).astype(np.float32)
def rgb8(rgb):return np.clip(rgb*255,0,255).astype(np.uint8)
def basic(rgb):
    g=cv2.cvtColor(rgb8(rgb),cv2.COLOR_RGB2GRAY);gf=g.astype(float)
    h=cv2.calcHist([np.ascontiguousarray(g)],[0],None,[256],[0,256]).ravel()
    p=h/h.sum();p=p[p>0]
    return {'entropy':float(-(p*np.log2(p)).sum()),'contrast':float(gf.std()/255),'sharpness':float(cv2.Laplacian(gf,cv2.CV_64F).var()/255**2)}
def candidates(rgb,gammas,models):
    inp=rgb.astype(np.float32)/255.0;rows=[];images=[]
    for g in gammas:
        im=gamma_float(inp,g);fe=basic(im);df=pd.DataFrame([fe]);images.append(rgb8(im))
        rows.append({**fe,'brightness':float(cv2.cvtColor(rgb8(im),cv2.COLOR_RGB2GRAY).mean()/255),**{k:float(predict(models[k],df,k=='nr_mono')[0]) for k in ['fuzzy','nr_mono']}})
    return {'images':images,'features':np.array([[r[k] for k in NR] for r in rows]),**{k:np.array([r[k] for r in rows]) for k in ['brightness','fuzzy','nr_mono']}}
def choose(r,gammas,method,previous):
    vals=-np.abs(r['brightness'][:9]-.45) if method=='brightness' else r['features'][:9,0] if method=='entropy' else r[method][:9]
    valid=np.isfinite(vals);idx=np.flatnonzero(valid&np.isclose(vals,vals[valid].max(),atol=1e-9,rtol=0)) if valid.any() else np.arange(9)
    return min([gammas[i] for i in idx],key=lambda g:(np.round(abs(g-previous),12),np.round(abs(g-2/3),12),g))
