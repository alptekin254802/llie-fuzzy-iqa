"""Headless webcam processing benchmark. A physical DSHOW webcam is required."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,time,datetime
from pathlib import Path
import cv2,numpy as np
from threadpoolctl import threadpool_limits
import camera,selftest,kernel
from lut_candidates import make_tables
cv2.setNumThreads(1);threadpool_limits(1)
class TimedRead:
    def __init__(self,cap):self.cap=cap
    def read(self):
        self.start=time.perf_counter_ns();value=self.cap.read();self.end=time.perf_counter_ns()
        return value

def state_check(path):
    if path and json.loads(path.read_text(encoding='utf-8-sig')).get('status')=='running':
        raise RuntimeError('External experiment state reports running; timing disabled.')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=2000);p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--exposure',type=int,choices=[-3,-4,-5,-6,-7],default=-6);p.add_argument('--index',type=int,default=0)
    p.add_argument('--external-state',type=Path,help='Optional JSON: status=running blocks timing.')
    a=p.parse_args()
    if not 0<a.warmup<a.frames:p.error('Require 0 < warmup < total frames')
    state_check(a.external_state);a.output.mkdir(parents=True,exist_ok=False)
    mods=camera.models(a.models);cap=camera.open_camera(a.exposure,a.index);rows=[]
    try:
        result=selftest.run((camera.resize(camera.read(cap)) for _ in range(100)),mods)
        (a.output/'selftest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        state_check(a.external_state)
        camera.validate_format(cap);camera.validate_controls(cap,a.exposure)
        timer=TimedRead(cap);last_process=None
        gs=selftest.GAMMAS;tables=make_tables(gs);previous=2/3;last=None
        for i in range(a.frames):
            state_check(a.external_state)
            t0=time.perf_counter_ns();im=camera.read(timer);t1=time.perf_counter_ns()
            enhanced,gamma=kernel.process_frame(im,mods,gs,tables,previous);t2=time.perf_counter_ns()
            rows.append({'frame_index':i,'warmup':i<a.warmup,'capture_start_ns':t0,'cap_read_start_ns':timer.start,'cap_read_end_ns':timer.end,'capture_decode_ms':(timer.end-timer.start)/1e6,'capture_to_output_ms':(t2-t0)/1e6,'end_to_end_loop_ms':(t2-last_process)/1e6 if last_process else None,'capture_end_ns':t1,'processing_end_ns':t2,'capture_wait_ms':(t1-t0)/1e6,'processing_ms':(t2-t1)/1e6,'camera_interval_ms':(t1-last)/1e6 if last else None,'gamma':float(gamma)})
            previous=gamma;last=t1;last_process=t2
        state_check(a.external_state)
        retained=rows[a.warmup:];x=np.array([r['processing_ms'] for r in retained]);interval=np.array([r['camera_interval_ms'] for r in retained])
        elapsed=(retained[-1]['processing_end_ns']-rows[a.warmup-1]['processing_end_ns'])/1e9
        report={'frames_total':len(rows),'retained':len(retained),'exposure':a.exposure,'display':False,'processing_ms':{'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())},'camera_interval_median_ms':float(np.median(interval)),'deadline_miss_count':int(np.sum(x>interval)),'deadline_miss_fraction':float(np.mean(x>interval)),'actual_loop_fps':len(retained)/elapsed,'camera_controls':camera.validate_controls(cap,a.exposure),'camera_format':camera.validate_format(cap)}
        (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    finally:
        cap.release()
        if rows:
            with (a.output/'frames.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
