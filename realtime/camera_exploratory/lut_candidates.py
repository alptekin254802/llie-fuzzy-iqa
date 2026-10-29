"""Exact uint8-domain candidate tables; float candidates stay separate for metrics."""
import numpy as np
import cv2
from optimized_loop import FEATURES,batch_predict

def make_tables(gammas):
    values=np.arange(256,dtype=np.uint8).astype(np.float32)/255.0
    tables=[]
    for g in gammas:
        # Literal original gamma kernel, including scalar type and both casts/clips.
        gamma=1/g
        im=np.power(values.astype(np.float32),1.0/gamma)
        im=np.clip(im,0.0,1.0).astype(np.float32)
        im=np.clip(im,0,1).astype(np.float32)
        tables.append(np.clip(im*255,0,255).astype(np.uint8))
    return tables

def features_rgb8(rgb):
    if rgb.dtype!=np.uint8:raise TypeError('uint8 RGB required')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gf=gray.astype(float)
    h=cv2.calcHist([np.ascontiguousarray(gray)],[0],None,[256],[0,256]).ravel()
    p=h/h.sum();p=p[p>0]
    fe={'entropy':float(-(p*np.log2(p)).sum()),'contrast':float(gf.std()/255),'sharpness':float(cv2.Laplacian(gf,cv2.CV_64F).var()/255**2)}
    return fe,float(gray.mean()/255)

def lut_candidates(rgb,tables,models,scorers=('fuzzy','nr_mono'),keep_images=False):
    if rgb.dtype!=np.uint8 or rgb.shape!=(256,256,3):raise ValueError('Expected 256x256 RGB uint8')
    features=[];brightness=[];images=[]
    for table in tables:
        im=cv2.LUT(rgb,table);fe,b=features_rgb8(im)
        features.append([fe[n] for n in FEATURES]);brightness.append(b)
        if keep_images:images.append(im)
    features=np.array(features,dtype=float)
    result={'features':features,'brightness':np.array(brightness),'images':images}
    for name in scorers:result[name]=batch_predict(models[name],features,name=='nr_mono')
    return result
