"""Paths, image loading and numerical IO for parameter selection."""
from pathlib import Path
import hashlib, json, zipfile
import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO/'results/studies/parameter_selection'
ROOTS = {}

def configure(path):
    global ROOTS
    p=Path(path).resolve()
    ROOTS={k:(Path(v) if Path(v).is_absolute() else p.parent/v).resolve() for k,v in json.loads(p.read_text())['data_roots'].items()}

def resolve(alias):
    parts=str(alias).replace('\\','/').split('/',1)
    if parts[0] not in ROOTS:raise KeyError('Configure dataset root: '+parts[0])
    return ROOTS[parts[0]]/(parts[1] if len(parts)==2 else '')

def read_rgb(row,reference=False):
    prefix='reference' if reference else 'input'
    if row.get(prefix+'_path',''):raw=resolve(row[prefix+'_path']).read_bytes()
    else:
        with zipfile.ZipFile(resolve(row['archive_path'])) as archive:raw=archive.read(row[prefix+'_member'])
    expected=row.get(prefix+'_sha256','')
    if expected and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Image hash mismatch: '+str(row['scene']))
    bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError('Unreadable image: '+str(row['scene']))
    return cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255

def read(path):return pd.read_csv(path,dtype={'scene':str},float_precision='round_trip')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n').replace('\n','\r\n').encode())
def csv(path,frame):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(frame).to_csv(p,index=False,lineterminator='\r\n')
def output_dir(path):
    p=Path(path).resolve()
    if p==REPO or REPO in p.parents:raise ValueError('Use a work directory outside the repository.')
    p.mkdir(parents=True,exist_ok=True)
    return p
