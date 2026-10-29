import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np
from threadpoolctl import threadpool_limits
from lut_candidates import make_tables,lut_candidates
from optimized_loop import select_gamma
import reference,camera
cv2.setNumThreads(1);threadpool_limits(1)
METHODS=['fuzzy','nr_mono','entropy','brightness']
GAMMAS=np.round(np.arange(.2,1.001,.1),1)
def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise AssertionError('Bit equality failed: '+label)
def run(frames,models):
    gammas=[*GAMMAS,2/3];tables=make_tables(gammas);previous={m:2/3 for m in METHODS};h1=hashlib.sha256();h2=hashlib.sha256();count=0
    for rgb in frames:
        a=reference.candidates(rgb,gammas,models);b=lut_candidates(rgb,tables,models,keep_images=True)
        for key in ['features','brightness','fuzzy','nr_mono']:
            exact(a[key],b[key],key);h1.update(a[key].tobytes());h2.update(b[key].tobytes())
        for i,(x,y) in enumerate(zip(a['images'],b['images'])):
            exact(x,y,'image '+str(i));h1.update(x.tobytes());h2.update(y.tobytes())
        grid={k:v[:9] for k,v in b.items()}
        for method in METHODS:
            for prev in [previous[method],2/3]:
                x=reference.choose(a,gammas,method,prev);y=select_gamma(grid,GAMMAS,method,prev);exact(x,y,'selection '+method)
            previous[method]=reference.choose(a,gammas,method,previous[method])
        count+=1
    return {'passed':True,'frames':count,'candidates':10*count,'original_sha256':h1.hexdigest(),'lut_sha256':h2.hexdigest(),'numpy':np.__version__,'opencv':cv2.__version__}
def main():
    p=argparse.ArgumentParser();p.add_argument('--models');p.add_argument('--frames',type=int,default=100);p.add_argument('--synthetic',action='store_true');p.add_argument('--exposure',type=int,default=-5);p.add_argument('--output',type=Path);a=p.parse_args();mods=camera.models(a.models);cap=None
    try:
        if a.synthetic:
            rng=np.random.default_rng(20261023);frames=(rng.integers(0,256,(256,256,3),dtype=np.uint8) for _ in range(a.frames))
        else:
            cap=camera.open_camera(a.exposure);frames=(camera.resize(camera.read(cap)) for _ in range(a.frames))
        result=run(frames,mods);result['synthetic']=a.synthetic
        if a.output:a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
    finally:
        if cap is not None:cap.release()
if __name__=='__main__':main()
