"""Verify released estimates and exact CLAHE kernels without GPU or a webcam."""
import argparse,itertools,json
import cv2,numpy as np,pandas as pd
from analysis.parameter_selection.common import DATA,REPO,sha,read
from analysis.parameter_selection.candidates import candidates,exposed,features
from analysis.parameter_selection.analyse import calculate,compare
from realtime.clahe.fast import Engine,PARAMS,choose
from realtime.clahe.core import Engine as NativeEngine,prep

def kernels():
    cv2.setNumThreads(1);engine=Engine();native=NativeEngine();rng=np.random.default_rng(20261030)
    samples=[np.zeros((256,256,3),np.float32),np.ones((256,256,3),np.float32),np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3)).astype(np.float32)/255,rng.random((256,256,3),dtype=np.float32)]
    for raw in samples:
        expected=list(itertools.islice(candidates(exposed(raw,1.)),9));actual=engine.inspect(raw)
        for i,(_,p,_,im) in enumerate(expected):
            assert p==PARAMS[i] and im.tobytes()==actual['candidate_float32'][i].tobytes()
            assert np.array_equal([features(im)[k] for k in engine.models['NR-free'].features],actual['features'][i])
        for selector in ['NR-free','NR-mono','Linear3']:
            x=pd.DataFrame(actual['features'],columns=['entropy','contrast','sharpness'])
            expected_scores=engine.models[selector].predict(x)
            assert np.allclose(expected_scores,actual['scores_by_selector'][selector],atol=1e-12,rtol=0)
            assert choose(expected_scores)==actual['selected_by_selector'][selector]['index']
    bgr=rng.integers(0,256,(480,640,3),dtype=np.uint8)
    for p in PARAMS:assert native.apply(bgr,p).tobytes()==native.reference(bgr,p).tobytes()
    return {'synthetic_inputs':len(samples),'candidate_pixels':len(samples)*9,'native_parameter_checks':9}

def manifest():
    p=DATA/'PUBLIC_EXPORT.json'
    if not p.exists():raise FileNotFoundError(p)
    j=json.loads(p.read_text())
    for name,h in j['files'].items():assert sha(REPO/name)==h,name
    return len(j['files'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--kernels-only',action='store_true');a=p.parse_args()
    print('kernels',kernels())
    if not a.kernels_only:
        print('manifest',manifest())
        for part in ['screen','confirmation']:print(part,compare(part,calculate(part)))
if __name__=='__main__':main()
