"""Batched candidate search using float32 RGB inputs.

The search preserves the floating-point candidates without LUT quantization.
Its singleton matrix axis preserves the original per-row 1x27 @ 27 reduction
order rather than using a single 9x27 contraction."""
import itertools
import numpy as np
import cv2
FEATURES=('entropy','contrast','sharpness')
COMBOS=list(itertools.product(range(3),repeat=3))
def gamma_image(inp,g):
    if inp.dtype!=np.float32: raise TypeError('Expected original float32 RGB input')
    gamma=1/g
    corrected=np.power(inp,1.0/gamma)
    # Same two clips and float32 casts as source; only redundant allocations removed.
    np.clip(corrected,0.0,1.0,out=corrected)
    corrected=corrected.astype(np.float32,copy=False)
    np.clip(corrected,0,1,out=corrected)
    return corrected.astype(np.float32,copy=False)
def features_brightness(im):
    rgb=np.clip(im*255,0,255).astype(np.uint8)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gf=gray.astype(float)
    h=cv2.calcHist([np.ascontiguousarray(gray)],[0],None,[256],[0,256]).ravel()
    p=h/h.sum();p=p[p>0]
    fe={'entropy':float(-(p*np.log2(p)).sum()),'contrast':float(gf.std()/255),'sharpness':float(cv2.Laplacian(gf,cv2.CV_64F).var()/255**2)}
    return fe,float(gray.mean()/255)
def trapmf(x,a,b,c,d):
    x=np.asarray(x,float)
    left=np.where(b>a,(x-a)/np.maximum(b-a,1e-12),1.0)
    right=np.where(d>c,(d-x)/np.maximum(d-c,1e-12),1.0)
    return np.clip(np.minimum(np.minimum(left,1.0),right),0,1)
def batch_predict(model,features,product=False):
    arr=[-features[:,i] if model['signs'][n] else features[:,i] for i,n in enumerate(FEATURES)]
    ms=[np.stack([trapmf(arr[n],*model['mfs'][n][j]) for j in range(3)],1) for n in range(3)]
    if product:
        fire=np.stack([ms[0][:,i]*ms[1][:,j]*ms[2][:,k] for i,j,k in COMBOS],1)
        design=fire/(fire.sum(1,keepdims=True)+1e-12)
        return np.matmul(design[:,None,:],np.array(model['theta'])).reshape(-1)
    fire=np.stack([np.minimum(np.minimum(ms[0][:,i],ms[1][:,j]),ms[2][:,k]) for i,j,k in COMBOS],1)
    den=fire.sum(axis=1)+1e-09
    return np.matmul(fire[:,None,:],np.array(model['theta'])).reshape(-1)/den
def optimized_candidates(inp,gammas,models,scorers=('fuzzy','nr_mono'),keep_images=False):
    features=[];brightness=[];images=[]
    for g in gammas:
        im=gamma_image(inp,g);fe,b=features_brightness(im)
        features.append([fe[n] for n in FEATURES]);brightness.append(b)
        if keep_images:images.append(im)
    features=np.array(features,dtype=float)
    result={'features':features,'brightness':np.array(brightness),'images':images}
    for name in scorers:result[name]=batch_predict(models[name],features,name=='nr_mono')
    return result
def select_gamma(result,gammas,method,prev):
    vals=-np.abs(result['brightness']-.45) if method=='brightness' else result['features'][:,0] if method=='entropy' else result[method]
    valid=np.isfinite(vals);vmax=vals[valid].max() if valid.any() else np.nan
    mask=valid&np.isclose(vals,vmax,atol=1e-09,rtol=0) if valid.any() else np.ones(len(vals),dtype=bool)
    choices=np.asarray(gammas)[mask]
    idx=np.lexsort((choices,np.round(np.abs(choices-2/3),12),np.round(np.abs(choices-prev),12)))[0]
    return choices[idx]
