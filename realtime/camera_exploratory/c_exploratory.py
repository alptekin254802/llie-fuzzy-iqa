"""Exploratory capture policy. The historical mathematical kernel stays unchanged."""
from c_common import *
import c_common as legacy

AMENDMENT3=json.loads((ROOT/'PLAN_AMENDMENT_3.json').read_text(encoding='utf-8'))
DATA=ROOT/'data'
DERIVED=ROOT/'derived'

def validate_exploratory():
    validate_plan()
    assert sha(ROOT/'PLAN_AMENDMENT_3.json')==(ROOT/'PLAN_AMENDMENT_3.sha256').read_text().split()[0]
    manifest=json.loads((ROOT/'PUBLIC_EXPORT.json').read_text(encoding='utf-8'))
    assert AMENDMENT3['base_plan_sha256']==manifest['original_sha256']['PLAN.json']
    assert AMENDMENT3['amendment_1_sha256']==manifest['original_sha256']['PLAN_AMENDMENT_1.json']
    assert AMENDMENT3['framework']['kind']=='exploratory'
    assert AMENDMENT3['framework']['p_values'] is False
    # Historical frozen-source hashes refer to the original private experiment.

def private_path(*parts):
    p=DATA.joinpath(*parts).resolve();p.relative_to(DATA.resolve());return p

def camera_info(cap,require=True):
    code=cap.get(cv2.CAP_PROP_FOURCC)
    info={'backend':cap.getBackendName(),'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),'fourcc_numeric':code,'fourcc_text':''.join(chr((int(code)>>(8*i))&255) for i in range(4)),'fps_readback':cap.get(cv2.CAP_PROP_FPS),'controls':readbacks(cap)}
    if require and (info['backend']!='DSHOW' or info['width']!=640 or info['height']!=480 or info['fourcc_text']!='YUY2'):
        raise RuntimeError('Could not verify YUY2/DSHOW/640x480: '+str(info))
    return info

def open_camera():
    validate_exploratory();cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():raise RuntimeError('Could not open camera; close other camera applications.')
        for name,value in [('FRAME_WIDTH',640),('FRAME_HEIGHT',480),('FPS',30),('FOURCC',cv2.VideoWriter_fourcc(*'YUY2')),*PLAN['capture']['controls'].items(),('EXPOSURE',-3)]:
            if not cap.set(getattr(cv2,'CAP_PROP_'+name),value):raise RuntimeError('Camera setting rejected: '+name)
        check_settings(cap,-3);camera_info(cap);return cap
    except Exception:cap.release();raise

def frame(cap):
    im=legacy.frame(cap);camera_info(cap);return im

def settle(cap,exposure):
    if not cap.set(cv2.CAP_PROP_EXPOSURE,exposure):raise RuntimeError('Exposure request rejected')
    for _ in range(PLAN['capture']['settle_discard_frames']):frame(cap)
    result=check_settings(cap,exposure);camera_info(cap);return result

def preview_metrics(images,exposure):
    assert 1<=len(images)<=5
    avg=np.mean(images,axis=0).astype(np.float32)
    lum=float(cv2.cvtColor(avg,cv2.COLOR_BGR2GRAY).mean()/255)
    clip=float(np.any(avg>=255,axis=2).mean())
    valid=len(images)==5 and .35<=lum<=.55 and clip<.01 and exposure-4>=-7
    return {'exposure':exposure,'candidate_E0':exposure if valid else None,'luminance':lum,'clipping':clip,'single_frame_clipping':float(np.any(images[-1]>=255,axis=2).mean()),'preview_frames':len(images),'ready':valid}

def validate_labels(room,direction):
    room=room.strip();direction=direction.strip()
    if not room or not direction:raise ValueError('Room and direction labels cannot be empty.')
    if any(c in room+direction for c in '\r\n\x00'):raise ValueError('Labels must fit on one line.')
    return room,direction

def validate_scene_metadata(meta,smoke=False):
    assert meta['plan_sha256']==sha(ROOT/'PLAN.json')
    assert meta['amendment_sha256']==sha(ROOT/'PLAN_AMENDMENT_1.json')
    assert meta['amendment_3_sha256']==sha(ROOT/'PLAN_AMENDMENT_3.json')
    assert meta['analysis_framework']=='exploratory'
    assert bool(meta['dry_run'])==smoke and bool(meta['excluded_from_all_analyses'])==smoke
    validate_labels(meta['room_label'],meta['direction_label'])
    assert meta['camera']['fourcc']=='YUY2'

def duplicate_direction(room,direction):
    for p in private_path('accepted').glob('*/metadata.json'):
        m=json.loads(p.read_text(encoding='utf-8'))
        if not m.get('dry_run') and m.get('qc',{}).get('passed') and (m.get('room_label'),m.get('direction_label'))==(room,direction):return m['scene_id']
    return None

def write_qc_record(meta):
    d=DERIVED/'qc';d.mkdir(parents=True,exist_ok=True)
    record={k:meta.get(k) for k in ['scene_id','room_label','direction_label','dry_run','excluded_from_all_analyses','analysis_framework','started_utc','finished_utc','amendment_3_sha256','E0','reference_exposure_probes','qc','error']}
    record['reference_exposure_probes']=[{k:v for k,v in x.items() if k!='raw_frames'} for x in meta['reference_exposure_probes']]
    record['frame_count']=len(meta['frames']);record['camera_format']='YUY2'
    save(d/(meta['scene_id']+'.json'),record)
