"""Exploratory YUY2 capture. Preview is never saved. Real capture needs Space."""
from c_exploratory import *
import argparse,unicodedata

def progress(cap, text):
    callback=getattr(cap,'progress',None)
    if callback is not None:callback(text)

def write_png(cap,folder,name,im,e):
    info=camera_info(cap);check_settings(cap,e);p=folder/name
    if not cv2.imwrite(str(p),im):raise RuntimeError('Could not write PNG')
    return {'file':name,'sha256':sha(p),'dtype':str(im.dtype),'shape':list(im.shape),'utc':now(),'readbacks':info['controls'],'camera_readback':info}

def reference(cap,folder,label,e):
    progress(cap,('Beginning' if label=='reference_begin' else 'Ending')+' reference: waiting for exposure to settle...')
    settle(cap,e);total=np.zeros((480,640,3),dtype=np.uint16);raw=[]
    for i in range(16):
        progress(cap,('Beginning' if label=='reference_begin' else 'Ending')+f' reference: frame {i+1}/16')
        im=frame(cap);total+=im.astype(np.uint16);rec=write_png(cap,folder,f'{label}_{i:02d}.png',im,e);rec['gray8_luminance']=luminance8(im);raw.append(rec)
    record=write_png(cap,folder,label+'_sum16.png',total,e);record.update(raw_frames=raw,scale=4080,metrics=reference_metrics(total));return total,record

def probe_e0(cap,folder=None):
    probes=[]
    for e in AMENDMENT['changes']['E0']['candidate_order']:
        progress(cap,f'E0 scan: exposure {e}; 12 settling and 5 check frames...')
        settle(cap,e);ims=[frame(cap) for _ in range(5)];probe=preview_metrics(ims,e);probe['camera_readback']=camera_info(cap)
        if folder is not None:probe['raw_frames']=[write_png(cap,folder,f'e0_probe_{e}_{i:02d}.png',im,e) for i,im in enumerate(ims)]
        probes.append(probe)
        if probe['ready']:return e,probes
    return None,probes

def acquire(cap,dry_run=False,scene_id='dry_run_3',room_label='',direction_label=''):
    validate_exploratory();room_label,direction_label=validate_labels(room_label,direction_label)
    if not dry_run and duplicate_direction(room_label,direction_label):raise ValueError('This room/direction label was already accepted; enter a new label for a new view.')
    name=scene_id if dry_run else datetime.datetime.now().strftime('scene_%Y%m%d_%H%M%S_%f')
    if dry_run and (not name.startswith('dry_run') or Path(name).name!=name):raise ValueError('Invalid dry-run name')
    if any(private_path(d,name).exists() for d in ['accepted','rejected','dry_run','pending']):raise RuntimeError('Scene already exists; no overwrite: '+name)
    folder=private_path('pending',name);folder.mkdir(parents=True)
    meta={'scene_id':name,'dry_run':dry_run,'excluded_from_all_analyses':dry_run,'analysis_framework':'exploratory','room_label':room_label,'direction_label':direction_label,'started_utc':now(),'plan_sha256':sha(ROOT/'PLAN.json'),'amendment_sha256':sha(ROOT/'PLAN_AMENDMENT_1.json'),'amendment_3_sha256':sha(ROOT/'PLAN_AMENDMENT_3.json'),'camera':{**PLAN['capture'],'fourcc':'YUY2','reference_exposure_candidates':AMENDMENT['changes']['E0']['candidate_order'],'E0_selection':'canonical order -3,-2,-1; luminance .35-.55, clipping <.01, E0-4 >=-7'},'frames':[],'reference_exposure_probes':[],'qc':{'passed':False,'reasons':['incomplete']}}
    try:
        meta['before_scene_readback']=camera_info(cap)
        e0,meta['reference_exposure_probes']=probe_e0(cap,folder)
        if e0 is None:raise RuntimeError('QC: no valid E0 (-3,-2,-1). Adjust indirect lighting or framing outside capture; keep thresholds unchanged.')
        meta['E0']=e0;begin,meta['reference_begin']=reference(cap,folder,'reference_begin',e0)
        for stop in [1,2,3,4]:
            progress(cap,f'Dark level {stop}/4: waiting for exposure to settle...')
            settle(cap,e0-stop)
            for j in range(5):
                progress(cap,f'Dark level {stop}/4: frame {j+1}/5')
                im=frame(cap);rec=write_png(cap,folder,f'dark_stop{stop}_{j:02d}.png',im,e0-stop);rec.update(order=len(meta['frames']),stop=stop,repetition=j,exposure=e0-stop,gray8_luminance=luminance8(im));meta['frames'].append(rec)
        end,meta['reference_end']=reference(cap,folder,'reference_end',e0)
        baseline=float(np.mean([r['gray8_luminance'] for r in meta['reference_begin']['raw_frames']]))
        means=[float(np.mean([r['gray8_luminance'] for r in meta['frames'] if r['stop']==s])) for s in [1,2,3,4]]
        progress(cap,'Calculating QC and completing the record...')
        meta['qc']=exposure_qc(qc(begin,end),baseline,means,e0)
    except Exception as exc:
        meta['error']=str(exc);meta['qc']={'passed':False,'reasons':[str(exc)]}
    finally:
        try:meta['final_camera_readback']=camera_info(cap)
        except Exception as exc:
            meta['final_camera_readback_error']=str(exc);meta['qc']['passed']=False;meta['qc']['reasons'].append('final_format_or_settings_unverified')
        meta['excluded_from_all_analyses']=dry_run or not meta['qc']['passed']
        meta['finished_utc']=now();save(folder/'metadata.json',meta)
        dest=private_path('dry_run' if dry_run else 'accepted' if meta['qc']['passed'] else 'rejected',name)
        dest.parent.mkdir(parents=True,exist_ok=True);folder.resolve().relative_to(DATA.resolve());dest.resolve().relative_to(DATA.resolve());folder.rename(dest)
        write_qc_record(meta)
    print(json.dumps({'scene':name,'folder':str(dest),'room_label':room_label,'direction_label':direction_label,'qc':meta['qc'],'dry_run':dry_run},ensure_ascii=False),flush=True)
    return dest,meta

def overlay(im,metrics,room,direction,message):
    out=im.copy();lines=[f'Room: {room} | Direction: {direction}',f'YUY2 | preview exposure {metrics["exposure"]} | E0: {metrics["candidate_E0"] if metrics["ready"] else "not valid"}',f'Luminance(avg5): {metrics["luminance"]:.3f} [0.35,0.55]',f'Clipped(avg5): {100*metrics["clipping"]:.2f}% (<1%) | current: {100*metrics["single_frame_clipping"]:.2f}%', '1:-3  2:-2  3:-1 | E: scan E0 | L: labels', 'SPACE: fresh scan + capture | Q/Esc: quit',message]
    for i,line in enumerate(lines):
        text=unicodedata.normalize('NFKD',line).encode('ascii','replace').decode('ascii')[:95]
        cv2.putText(out,text,(8,22+23*i),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,0),3)
        cv2.putText(out,text,(8,22+23*i),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,255,0) if metrics['ready'] else (0,220,255),1)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--scene-id',default='dry_run_3');p.add_argument('--auto',action='store_true');p.add_argument('--no-preview',action='store_true');p.add_argument('--room',default='');p.add_argument('--direction',default='');a=p.parse_args()
    if a.auto and not a.dry_run:p.error('--auto is only allowed for dry-run')
    if a.no_preview and not a.auto:p.error('--no-preview requires --auto --dry-run')
    if a.auto:
        room,direction=validate_labels(a.room or 'dry_run_room',a.direction or 'dry_run_direction')
        cap=open_camera()
        try:acquire(cap,True,a.scene_id,room,direction)
        finally:cap.release()
        return
    from capture_gui import run_gui
    run_gui(a,acquire,probe_e0)
if __name__=='__main__':main()
