"""Serial selection at 256 square followed by native-resolution CLAHE."""
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import datetime,json,time,platform,subprocess
import numpy as np,pandas as pd,cv2,psutil
from threadpoolctl import threadpool_limits,threadpool_info
from analysis.parameter_selection.common import REPO,DATA,read,read_rgb,resolve,sha,save,csv as writecsv
from analysis.parameter_selection.candidates import exposed,apply_clahe
from .fast import Engine as SerialEngine,PARAMS
threadpool_limits(1);cv2.setNumThreads(1)
ARCHIVE=REPO/'results/studies/clahe_live'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def prep(bgr):return cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.

class Engine:
    def __init__(self):
        self.serial=SerialEngine()
        v=np.arange(256,dtype=np.uint8)
        self.lut=np.clip(exposed(v.astype(np.float32)/255.,1.)*255.,0.,255.).astype(np.uint8)
        self.native={float(p):cv2.createCLAHE(clipLimit=float(p),tileGridSize=(8,8)) for p in PARAMS[1:]}
    def apply(self,bgr,param):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pixels=cv2.LUT(rgb,self.lut)
        if param==-1:return cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
        lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB);light,a,b=cv2.split(lab)
        rgb=cv2.cvtColor(cv2.merge([self.native[param].apply(light),a,b]),cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    def process(self,bgr,native=True):
        param,small=self.serial.search(prep(bgr),'NR-mono')
        return (self.apply(bgr,param) if native else small),param
    def reference(self,bgr,param):
        low=exposed(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.,1.)
        im=low.copy() if param==-1 else apply_clahe(low,clip_limit=param,tile_grid_size=(8,8))
        return cv2.cvtColor(np.clip(im*255.,0.,255.).astype(np.uint8),cv2.COLOR_RGB2BGR)

def gamma_setup():
    # The existing measured gamma kernel uses sibling imports.
    sys.path.insert(0,str(REPO/'realtime'))
    import kernel,lut_candidates
    models={k:json.loads((REPO/'models'/n).read_text()) for k,n in [('fuzzy','nr_free.json'),('nr_mono','nr_mono.json')]}
    gammas=np.asarray(json.loads((ARCHIVE/'CAMERA.json').read_text())['gammas'],dtype=np.float64)
    return kernel,models,gammas,lut_candidates.make_tables(gammas)

def load_inputs():
    sample=read(DATA/'timing_inputs.csv').fillna('');out={'600x400':[],'640x480':[]}
    for row in sample.to_dict('records'):
        im=cv2.imread(str(resolve(row['native_input_path'])),cv2.IMREAD_COLOR)
        if im is None or im.shape!=(400,600,3):raise ValueError('Expected native LOL 600x400 image')
        out['600x400'].append(im);out['640x480'].append(cv2.resize(im,(640,480),interpolation=cv2.INTER_LINEAR))
    return sample,out

def check_freeze():
    path=DATA/'PUBLIC_EXPORT.json';j=json.loads(path.read_text())
    for name,h in j['files'].items():
        if sha(REPO/name)!=h:raise ValueError('Release file changed: '+name)
    return sha(path)

def environment():
    power=subprocess.run(['powercfg','/getactivescheme'],capture_output=True,text=True).stdout.strip() if os.name=='nt' else 'unavailable'
    return dict(pid=os.getpid(),python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,cv2_threads=cv2.getNumThreads(),threadpools=threadpool_info(),power=power)

def load_check(folder,label):
    cpu=psutil.cpu_percent(interval=5)
    result=dict(utc=now(),cpu_percent=cpu,passed=cpu<=10)
    save(folder/(label+'.json'),result)
    print('LOAD',label,cpu,'PASS' if result['passed'] else 'PENDING',flush=True)
    return result['passed']
