"""Run the fixed Q-Align comparison described in docs/QALIGN.md."""
from pathlib import Path
import os,sys,json,csv,time,datetime,hashlib,subprocess,traceback,argparse
from analysis.studies.config import OUT,REPO,HUMAN_ROOTS
O=OUT;R=OUT.parent
DS=['squarelol','lieq','rlie'];COUNTS={'squarelol':(2900,290),'lieq':(1000,100),'rlie':(1540,154)}
STATE=O/'qalign_run.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def state(**kwargs):
 d=json.loads(STATE.read_text(encoding='utf-8'));d.update(kwargs);d['updated_utc']=now();save(STATE,d)
def prepare():
 import numpy as np,pandas as pd
 from analysis.studies import common as c
 frozen=json.loads((REPO/'models/nr_free.json').read_text());rows=[]
 folders={'squarelol':HUMAN_ROOTS['squarelol']/'enhancement','lieq':HUMAN_ROOTS['lieq']/'dataset/Enhancement','rlie':HUMAN_ROOTS['rlie']/'enhancement'}
 baseline_path=REPO/'results/studies/qalign_nr_free.csv'
 baseline=pd.read_csv(baseline_path,dtype={'scene':str},float_precision='round_trip') if baseline_path.exists() else None
 for ds in DS:
  x=pd.read_csv(O/f'feat_{ds}_area256.csv',dtype={'scene':str}).sort_values('name')
  assert not x.name.duplicated().any() and (len(x),x.scene.nunique())==COUNTS[ds]
  pred=c.fuzzy_predict(x,frozen['signs'],frozen['mfs'],np.asarray(frozen['theta']))
  x=x.assign(nr_free=pred)
  if baseline is not None:
   z=x.merge(baseline[baseline.dataset==ds][['name','nr_free']],on='name',validate='one_to_one',suffixes=('','_recorded'))
   assert len(z)==len(x) and np.max(abs(z.nr_free-z.nr_free_recorded))<1e-10
   x=z.drop(columns='nr_free').rename(columns={'nr_free_recorded':'nr_free'})
  for row in x.itertuples():
   path=folders[ds]/row.name;assert path.exists(),path
   rows.append({'dataset':ds,'name':row.name,'scene':row.scene,'mos':float(row.mos),'nr_free':float(row.nr_free),'path':str(path)})
 pd.DataFrame(rows).to_csv(O/'qalign_manifest.csv',index=False)
 (O/'qalign_PLAN.md').write_bytes((REPO/'docs/QALIGN.md').read_bytes())
 print('Prepared fixed human-data manifest; no models fitted.',flush=True)

def worker():
 os.environ['HF_HUB_CACHE']=str(O/'qalign_hf_cache')
 os.environ['HUGGINGFACE_HUB_CACHE']=os.environ['HF_HUB_CACHE']
 os.environ['TRANSFORMERS_CACHE']=os.environ['HF_HUB_CACHE']
 from huggingface_hub import snapshot_download
 revision='dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb'
 snapshot_download('q-future/one-align',revision=revision,cache_dir=os.environ['HF_HUB_CACHE'],allow_patterns=['*.json','*.model','*.bin','*.txt'])
 expected={'pytorch_model-00001-of-00002.bin':'ca4cdc96237832a9eb3486274de0b10fa7235d5007ee6cff49320c7250970fe3','pytorch_model-00002-of-00002.bin':'d445a50608f025a39ac75c6da13cb9d12a4cc923f6e1b9abc4b767408eac0729'}
 snapshot=Path(os.environ['HF_HUB_CACHE'])/'models--q-future--one-align/snapshots'/revision
 for name,digest in expected.items():
  h=hashlib.sha256()
  with (snapshot/name).open('rb') as handle:
   for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==digest,name
 refs=Path(os.environ['HF_HUB_CACHE'])/'models--q-future--one-align/refs';refs.mkdir(parents=True,exist_ok=True);(refs/'main').write_text(revision)
 save(O/'qalign_model_source.json',{'sha':revision,'repo_id':'q-future/one-align','verified_weight_sha256':expected})
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
 state(status='loading',worker_pid=os.getpid(),python=sys.executable)
 print('IMPORT '+now(),flush=True)
 import numpy as np,pandas as pd,cv2,torch,pyiqa,bitsandbytes as bnb,importlib.metadata as meta
 cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.manual_seed(20261018);np.random.seed(20261018)
 versions={k:meta.version(k) for k in ['torch','torchvision','pyiqa','transformers','accelerate','bitsandbytes','huggingface-hub','tokenizers','numpy','opencv-python']}
 impl=Path(pyiqa.__file__).parent
 hashes={str(f.relative_to(impl)):sha(f) for f in [impl/'archs/qalign_arch.py',*sorted((impl/'archs/q_align').glob('*.py'))]}
 env={'versions':versions,'implementation_sha256':hashes,'gpu':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability(),'free_total_before_load_bytes':torch.cuda.mem_get_info(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu','--format=csv']).decode().strip(),'model_revision':json.loads((O/'qalign_model_source.json').read_text())['sha']}
 save(O/'qalign_environment.json',env)
 print('LOAD '+now(),flush=True);start=time.perf_counter();metric=pyiqa.create_metric('qalign_4bit',device='cuda');metric.eval()
 model=metric.net.model
 assert bool(getattr(model,'is_loaded_in_4bit',False)),'Refuse silent fp16 fallback'
 device_map=getattr(model,'hf_device_map',{})
 assert not any(str(v) in ['cpu','disk'] for v in device_map.values()),device_map
 assert all(p.device.type=='cuda' for p in model.parameters()),'CPU offload prohibited'
 env.update({'model_load_seconds':time.perf_counter()-start,'device_map':{k:str(v) for k,v in device_map.items()},'quantization_config':model.config.quantization_config.to_dict() if hasattr(model.config.quantization_config,'to_dict') else model.config.quantization_config,'internal_processor':metric.net.image_processor.to_dict(),'linear4bit_modules':sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),'memory_after_load':torch.cuda.mem_get_info()})
 save(O/'qalign_environment.json',env)
 data=pd.read_csv(O/'qalign_manifest.csv',dtype={'scene':str})
 def tensor(path):
  t0=time.perf_counter();bgr=cv2.imread(path,cv2.IMREAD_COLOR);assert bgr is not None,path
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
  h=hashlib.sha256(rgb.tobytes()).hexdigest();x=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
  prep=(time.perf_counter()-t0)*1000;t0=time.perf_counter();x=x.to('cuda');torch.cuda.synchronize();return x,prep,(time.perf_counter()-t0)*1000,h
 with torch.inference_mode():
  pilot=[]
  for ds in DS:
   for row in data[data.dataset==ds].head(3).itertuples():
    x,_,_,_=tensor(row.path);torch.cuda.synchronize();t0=time.perf_counter();v=float(metric(x,task_='quality',input_='image').item());torch.cuda.synchronize()
    assert np.isfinite(v) and 0<=v<=6,v
    if not pilot:state(status='scoring',setup_succeeded_utc=now(),first_valid_score_utc=now())
    pilot.append({'dataset':ds,'name':row.name,'score':v,'ms':(time.perf_counter()-t0)*1000})
  save(O/'qalign_pilot.json',{'warmup_first_three_excluded_from_timing':True,'calls':pilot})
  dest=O/'qalign_scores.csv';done=set()
  if dest.exists():
   prior=pd.read_csv(dest);assert not prior.duplicated(['dataset','name']).any();done=set(zip(prior.dataset,prior.name))
  fields=['dataset','name','scene','score','prepare_ms','h2d_ms','inference_ms','rgb256_sha256','cuda_allocated_bytes','cuda_reserved_bytes']
  with dest.open('a',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=fields)
   if not done:writer.writeheader();f.flush()
   for row in data.itertuples():
    if (row.dataset,row.name) in done:continue
    x,prep,h2d,h=tensor(row.path);torch.cuda.synchronize();t0=time.perf_counter();v=float(metric(x,task_='quality',input_='image').item());torch.cuda.synchronize();ms=(time.perf_counter()-t0)*1000
    assert np.isfinite(v),v
    writer.writerow(dict(dataset=row.dataset,name=row.name,scene=row.scene,score=v,prepare_ms=prep,h2d_ms=h2d,inference_ms=ms,rgb256_sha256=h,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved()));f.flush();done.add((row.dataset,row.name))
    if len(done)%10==0:state(status='scoring',completed=len(done),total=len(data),last_inference_ms=ms);print(f'{now()} {len(done)}/{len(data)} {ms:.1f} ms',flush=True)
  env.update({'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()});save(O/'qalign_environment.json',env)
 state(status='complete',completed=len(data),finished_utc=now(),scores_sha256=sha(O/'qalign_scores.csv'))
def supervise():
 deadline=datetime.datetime.fromisoformat(json.loads(STATE.read_text())['setup_deadline_utc'])
 with (O/'qalign_inference.log').open('a',encoding='utf-8') as log:
  proc=subprocess.Popen([sys.executable,'-u','-X','utf8','-B','-m','analysis.studies.qalign','worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   d=json.loads(STATE.read_text());now_dt=datetime.datetime.now(datetime.timezone.utc)
   if 'setup_succeeded_utc' not in d and now_dt>=deadline:
    proc.terminate();proc.wait(timeout=30);state(status='setup_timeout',finished_utc=now());print('STOP setup deadline',flush=True);raise SystemExit(124)
   time.sleep(1)
  print('Inference worker exit',proc.returncode,flush=True)
  if proc.returncode:raise SystemExit(proc.returncode)
def main():
 command=sys.argv[1]
 if command=='prepare':prepare();return
 if command=='analyze':
  from analysis.studies.qalign_summary import main as analyze
  analyze();return
 if command=='run':
  if STATE.exists() and json.loads(STATE.read_text()).get('status')=='complete':
   print('Already complete; use analyze.');return
  start=datetime.datetime.now(datetime.timezone.utc)
  previous=json.loads(STATE.read_text()) if STATE.exists() else None
  save(STATE,{'status':'starting','setup_started_utc':start.isoformat(),'setup_deadline_utc':(start+datetime.timedelta(hours=2)).isoformat(),'plan_sha256':sha(O/'qalign_PLAN.md'),'previous_attempt':previous})
  supervise();return
 if command=='worker':worker();return
 raise ValueError('Expected prepare, run, or analyze')
if __name__=='__main__':
 try:main()
 except Exception:
  error=traceback.format_exc()
  if STATE.exists():state(status='error',error=error,finished_utc=now())
  raise
