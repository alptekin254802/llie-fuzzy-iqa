import json,hashlib
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
PROFILE=json.loads((HERE/'camera_profile.json').read_text(encoding='utf-8'))
def models(directory=None):
    directory=Path(directory) if directory else HERE.parent/'models'
    result={}
    for key,name in [('fuzzy','nr_free'),('nr_mono','nr_mono')]:
        path=directory/(name+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=PROFILE['model_sha256'][name]:raise RuntimeError('Published model checksum mismatch: '+str(path))
        result[key]=json.loads(path.read_text(encoding='utf-8'))
    return result
def read(cap):
    ok,im=cap.read()
    if not ok or im is None or im.shape!=(480,640,3):raise RuntimeError('Webcam read failed or resolution differs from 640x480')
    return im
def resize(bgr):return cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(256,256),interpolation=cv2.INTER_LINEAR)
def validate_controls(cap,exposure):
    values={p:float(cap.get(getattr(cv2,'CAP_PROP_'+p))) for p in [*PROFILE['controls'],'EXPOSURE']}
    for p,v in PROFILE['controls'].items():
        if p=='AUTO_EXPOSURE':continue # DSHOW returns -1; manual behavior was image-validated.
        expected=2 if p=='AUTOFOCUS' else v
        if values[p]!=expected:raise RuntimeError(f'Camera control mismatch: {p}={values[p]}, expected {expected}')
    if values['EXPOSURE']!=exposure:raise RuntimeError('Exposure readback mismatch')
    return values
def validate_format(cap):
    code=int(cap.get(cv2.CAP_PROP_FOURCC))
    values={'backend':cap.getBackendName(),'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),'fps':cap.get(cv2.CAP_PROP_FPS),
            'fourcc':''.join(chr((code>>(8*i))&255) for i in range(4))}
    if (values['backend']!='DSHOW' or values['width']!=640 or values['height']!=480
            or abs(values['fps']-30)>.01 or values['fourcc']!='MJPG'):
        raise RuntimeError('Camera format readback mismatch: '+str(values))
    return values

def open_camera(exposure=-6,index=0):
    cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():raise RuntimeError('DSHOW webcam required. Close other camera applications.')
        requests=[('FRAME_WIDTH',640),('FRAME_HEIGHT',480),('FPS',30),
                  ('FOURCC',cv2.VideoWriter_fourcc('M','J','P','G')),
                  *PROFILE['controls'].items(),('EXPOSURE',exposure)]
        for name,value in requests:
            if not cap.set(getattr(cv2,'CAP_PROP_'+name),value):
                raise RuntimeError('Camera rejected setting: '+name)
        validate_format(cap)
        for _ in range(12):read(cap)
        validate_controls(cap,exposure)
        validate_format(cap)
        return cap
    except Exception:cap.release();raise
