"""Exact per-family candidate generation; no unrelated-family work."""
import sys
from pathlib import Path
import numpy as np,cv2
from analysis.parameter_selection import candidates as original


def finish(image):return np.ascontiguousarray(np.clip(image,0,1),dtype=np.float32)

def family_candidates(low,family):
    values=original.FAMILIES[family];result=[]
    if family=='clahe':
        # Identical quantisation and colour conversions, shared across clip limits.
        pixels=np.clip(low.astype(np.float32)*255.,0.,255.).astype(np.uint8)
        lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB);light,a,b=cv2.split(lab)
        for value in values:
            if value==-1:image=low.copy()
            else:
                enhanced=cv2.createCLAHE(clipLimit=float(value),tileGridSize=(8,8)).apply(light)
                rgb=cv2.cvtColor(cv2.merge([enhanced,a,b]),cv2.COLOR_LAB2RGB)
                image=np.clip(rgb.astype(np.float32)/255.,0.,1.).astype(np.float32)
            result.append((float(value),finish(image)))
    elif family=='he_blend':
        he=original.histogram_equalization(low)
        for value in values:result.append((float(value),finish(np.float32(1-value)*low+np.float32(value)*he)))
    elif family=='denoise':
        gamma=original.apply_gamma_correction(low,gamma=1.5)
        for value in values:
            image=gamma.copy() if value==0 else cv2.bilateralFilter(gamma,d=5,sigmaColor=float(value)/255.,sigmaSpace=2.,borderType=cv2.BORDER_REFLECT_101)
            result.append((float(value),finish(image)))
    elif family=='log':
        for value in values:
            image=low.copy() if value==0 else np.log1p(np.float32(value)*low)/np.log1p(np.float32(value))
            result.append((float(value),finish(image)))
    else:raise ValueError(family)
    return result

def single_candidate(low,family,value):
    assert value in original.FAMILIES[family]
    if family=='clahe':image=low.copy() if value==-1 else original.apply_clahe(low,clip_limit=value,tile_grid_size=(8,8))
    elif family=='log':image=low.copy() if value==0 else np.log1p(np.float32(value)*low)/np.log1p(np.float32(value))
    elif family=='he_blend':
        image=low.copy() if value==0 else np.float32(1-value)*low+np.float32(value)*original.histogram_equalization(low)
    elif family=='denoise':
        gamma=original.apply_gamma_correction(low,gamma=1.5)
        image=gamma.copy() if value==0 else cv2.bilateralFilter(gamma,d=5,sigmaColor=float(value)/255.,sigmaSpace=2.,borderType=cv2.BORDER_REFLECT_101)
    else:raise ValueError(family)
    return finish(image)

def required_features(image,names):
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);out={}
    if 'brightness' in names:out['brightness']=float(gray.mean()/255)
    if 'entropy' in names:
        hist=cv2.calcHist([np.ascontiguousarray(gray)],[0],None,[256],[0,256]).ravel();p=hist/hist.sum();p=p[p>0]
        out['entropy']=float(-(p*np.log2(p)).sum())
    if 'contrast' in names or 'sharpness' in names:
        g=gray.astype(float)
        if 'contrast' in names:out['contrast']=float(g.std()/255)
        if 'sharpness' in names:out['sharpness']=float(cv2.Laplacian(g,cv2.CV_64F).var()/255**2)
    return out
