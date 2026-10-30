from .core import *
from analysis.parameter_selection.common import output_dir,configure
import argparse
def main(number):
 freeze=check_freeze()
 folder=R/'offline'/f'session{number}'
 if folder.exists():raise RuntimeError('Do not replace a session')
 folder.mkdir(parents=True)
 e=Engine();sample,allimages=load_inputs();images=allimages['640x480'];kernel,mods,gammas,tables=gamma_setup();previous=2/3
 def invoke(kind,im):
  nonlocal previous
  if kind=='gamma':
   out,p=kernel.process_frame(im,mods,gammas,tables,previous);previous=p;return out,p
  return e.process(im,native=kind=='native')
 cases=['selection256','native','gamma']
 for k in cases:
  for im in images[:20]:invoke(k,im)
 if not load_check(folder,'before'):
  save(folder/'status.json',dict(status='PENDING',utc=now()));return 3
 previous=2/3;rows=[];rng=np.random.default_rng(20261101+number);start=now()
 for repeat in range(3):
  for i in rng.permutation(100):
   for kind in rng.permutation(cases):
    t=time.perf_counter_ns();out,p=invoke(kind,images[int(i)]);end=time.perf_counter_ns()
    rows.append(dict(session=number,repeat=repeat,scene=sample.scene.iloc[i],kind=kind,param=p,milliseconds=(end-t)/1e6))
 writecsv(folder/'frames.csv',rows)
 save(folder/'status.json',dict(status='COMPLETE',start_utc=start,end_utc=now(),calls=len(rows),environment=environment(),raw_sha256=sha(folder/'frames.csv'),freeze_sha256=freeze))
 print('OFFLINE_COMPLETE',number,len(rows),flush=True)
 return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();configure(a.config);R=output_dir(a.output);raise SystemExit(main(a.session))
