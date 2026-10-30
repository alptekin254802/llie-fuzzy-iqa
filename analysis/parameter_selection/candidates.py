"""Candidate pixels and features used in the parameter-selection experiments."""
import hashlib
import numpy as np
import cv2
from preprocessing.enhancement.methods import apply_clahe, apply_gamma_correction, apply_log_transform, histogram_equalization
from evaluation.metrics import compute_psnr, compute_ssim
from .common import read_rgb

def exposed(low,level):
    linear=np.where(low<=.04045,low/12.92,((low+.055)/1.055)**2.4)*level
    return np.clip(np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055),0,1).astype(np.float32)

def features(im):
    gray=cv2.cvtColor(np.clip(im*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);g=gray.astype(float)
    h=cv2.calcHist([np.ascontiguousarray(gray)],[0],None,[256],[0,256]).ravel();p=h/h.sum();p=p[p>0]
    return dict(brightness=float(gray.mean()/255),entropy=float(-(p*np.log2(p)).sum()),contrast=float(g.std()/255),sharpness=float(cv2.Laplacian(g,cv2.CV_64F).var()/255**2))
FAMILIES={'clahe':[-1,.5,1,1.5,2,3,4,6,8],'log':[0,1,2,4,8,16,32,64,128],'he_blend':[0,.125,.25,.375,.5,.625,.75,.875,1],'denoise':[0,5,10,15,25,40]}
FEATURES=['brightness','entropy','contrast','sharpness']
KEYS=['dataset','scene','reference_group','family','param','candidate_id']
def cid(family,param):return family+':'+format(float(param),'.12g')
def read_input(row):return exposed(read_rgb(row),1.)
def read_reference(row):return read_rgb(row,True)
def candidates(low):
    assert low.shape==(256,256,3) and low.dtype==np.float32 and np.isfinite(low).all() and low.min()>=0 and low.max()<=1
    he=histogram_equalization(low);gamma=apply_gamma_correction(low,gamma=1.5)
    for family,values in FAMILIES.items():
        for value in values:
            if family=='clahe':im=low.copy() if value==-1 else apply_clahe(low,clip_limit=value,tile_grid_size=(8,8))
            elif family=='log':im=low.copy() if value==0 else np.log1p(np.float32(value)*low)/np.log1p(np.float32(value))
            elif family=='he_blend':im=(np.float32(1-value)*low+np.float32(value)*he)
            else:im=gamma.copy() if value==0 else cv2.bilateralFilter(gamma,d=5,sigmaColor=float(value)/255.,sigmaSpace=2.,borderType=cv2.BORDER_REFLECT_101)
            im=np.ascontiguousarray(np.clip(im,0,1),dtype=np.float32)
            yield family,float(value),cid(family,value),im
    yield 'control',0.,'control:rawinput',low.copy()
    yield 'control',1.,'control:original_log',np.ascontiguousarray(apply_log_transform(low,c=1.),dtype=np.float32)
def pixel_hash(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
