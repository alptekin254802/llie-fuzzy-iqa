from .core import *
from analysis.parameter_selection.common import output_dir
import argparse
P=json.loads((ARCHIVE/'CAMERA.json').read_text())
def settings(cap):
 values={n:float(cap.get(getattr(cv2,'CAP_PROP_'+n))) for n in [*P['camera']['controls'],'EXPOSURE','FRAME_WIDTH','FRAME_HEIGHT','FPS','FOURCC']}
 values['backend']=cap.getBackendName()
 assert values['backend']=='DSHOW' and values['FRAME_WIDTH']==640 and values['FRAME_HEIGHT']==480 and abs(values['FPS']-30)<.01
 assert int(values['FOURCC'])==cv2.VideoWriter_fourcc(*'MJPG')
 for n,v in P['camera']['controls'].items():
  expected=-1 if n=='AUTO_EXPOSURE' else 2 if n=='AUTOFOCUS' else v
  assert values[n]==expected,(n,values[n],expected)
 assert values['EXPOSURE']==-6
 return values
def frame(cap):
 ok,im=cap.read()
 if not ok or im is None or im.shape!=(480,640,3):raise RuntimeError('Camera frame unavailable or wrong size')
 return im
def main(number,kind):
 freeze=check_freeze();folder=R/'live'/f'round{number}_{kind}'
 if folder.exists():raise RuntimeError('Do not replace a session')
 folder.mkdir(parents=True);cap=None;rows=[];state=dict(status='PREPARING',kind=kind,round=number,utc=now(),freeze_sha256=freeze)
 save(folder/'status.json',state)
 try:
  e=Engine();kernel,mods,gammas,tables=gamma_setup()
  if not load_check(folder,'before_camera'):state['status']='PENDING';return 3
  cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
  if not cap.isOpened():raise RuntimeError('DSHOW camera unavailable')
  requests=[]
  for name,value in [('FRAME_WIDTH',640),('FRAME_HEIGHT',480),('FPS',30),('FOURCC',cv2.VideoWriter_fourcc(*'MJPG')),*P['camera']['controls'].items(),('EXPOSURE',-6)]:
   ok=bool(cap.set(getattr(cv2,'CAP_PROP_'+name),value));requests.append(dict(name=name,value=value,accepted=ok))
   if not ok:raise RuntimeError('Camera request failed: '+name)
  state['requests']=requests;state['readback_initial']=settings(cap)
  for _ in range(12):frame(cap)
  state['readback_settled']=settings(cap)
  # Untimed native-output equality, no frames or hashes of private content retained.
  for _ in range(20):
   im=frame(cap);out,p=e.process(im);assert out.tobytes()==e.reference(im,p).tobytes()
  state['selftest_frames']=20;state['readback_selftest']=settings(cap)
  if not load_check(folder,'before_timing'):state['status']='PENDING';return 3
  state['readback_before_timing']=settings(cap);state['environment']=environment();state['start_utc']=now();state['status']='MEASURING';save(folder/'status.json',state)
  prev=2/3;last_capture=last_end=None
  for i in range(2000):
   t0=time.perf_counter_ns();im=frame(cap);t1=time.perf_counter_ns()
   if kind=='capture':out=im;p=None
   elif kind=='gamma':out,p=kernel.process_frame(im,mods,gammas,tables,prev);prev=p
   else:out,p=e.process(im)
   t2=time.perf_counter_ns()
   rows.append(dict(frame=i,warmup=i<50,capture_start_ns=t0,capture_end_ns=t1,output_end_ns=t2,capture_ms=(t1-t0)/1e6,processing_ms=(t2-t1)/1e6,camera_interval_ms=(t1-last_capture)/1e6 if last_capture else None,end_to_end_loop_ms=(t2-last_end)/1e6 if last_end else None,capture_to_output_ms=(t2-t0)/1e6,param=p))
   last_capture=t1;last_end=t2
   if (i+1)%500==0:print('LIVE_PROGRESS',number,kind,i+1,flush=True)
  state['readback_final']=settings(cap);state['cpu_after']=psutil.cpu_percent(interval=.2);state['status']='COMPLETE';state['end_utc']=now()
 except Exception as exc:
  state.update(status='FAILED',error=repr(exc));raise
 finally:
  if cap is not None:cap.release()
  if rows:writecsv(folder/'frames.csv',rows);state['raw_sha256']=sha(folder/'frames.csv')
  state.update(rows=len(rows),finished_utc=now());save(folder/'status.json',state)
 print('LIVE_COMPLETE',number,kind,flush=True);return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);p.add_argument('--kind',choices=['capture','gamma','clahe'],required=True);p.add_argument('--output',required=True);a=p.parse_args();R=output_dir(a.output);raise SystemExit(main(a.round,a.kind))
