"""Q-Align fp16 reproduction: unchanged official scoring, automatic GPU/CPU placement."""
from pathlib import Path
import os,sys,json,csv,time,datetime,hashlib,subprocess,traceback,ctypes,shutil,ast,contextlib
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--fourbit-scores',type=Path,required=True)
parser.add_argument('--fourbit-environment',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--cache-dir',type=Path,required=True,help='Hugging Face cache; pinned snapshot downloaded if absent')
args=parser.parse_args()
O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
for k in ['HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE']:os.environ[k]=str(args.cache_dir.resolve())
os.environ['HF_HOME']=str(O/'hf_home');os.environ['HF_MODULES_CACHE']=str(O/'hf_modules')
os.environ['TORCH_HOME']=str(O/'torch_cache');os.environ['MPLCONFIGDIR']=str(O/'mpl_cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'

STATE=O/'run.json';REV='dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def state(**kw):
 d=json.loads(STATE.read_text('utf-8')) if STATE.exists() else {};d.update(kw);d['updated_utc']=now();save(STATE,d)
def memory():
 import psutil
 return psutil.virtual_memory().available

def worker():
 # Atomic lock directory prevents concurrent writers on Windows and Linux.
 lock=O/'worker.lockdir'
 try:lock.mkdir()
 except FileExistsError:raise RuntimeError('Worker lock exists. Confirm no worker is active before removing the lock directory.')
 try:run()
 finally:lock.rmdir()


def run():
 existing=json.loads(STATE.read_text('utf-8')) if STATE.exists() else {}
 if existing.get('status')=='complete':print('Already complete.');return
 if existing.get('status')=='stopped_nonfinite':raise RuntimeError('Nonfinite stop is final; no automatic restart')
 avail=memory();state(status='preflight',worker_pid=os.getpid(),python=sys.executable,manifest_sha256=sha(args.manifest),free_ram_before_load_bytes=avail)
 if avail<16*2**30:
  state(status='stopped_ram',reason='Available physical RAM below16GiB before load');print('STOP RAM',avail/2**30,flush=True);return
 state(status='loading',load_started_utc=now())
 print('IMPORT',now(),flush=True)
 import numpy as np,pandas as pd,cv2,torch,pyiqa,importlib.metadata as meta
 from scipy.stats import spearmanr,pearsonr
 from huggingface_hub import snapshot_download
 snapshot=Path(snapshot_download('q-future/one-align',revision=REV,cache_dir=str(args.cache_dir.resolve()),allow_patterns=['*.json','*.model','*.bin','*.txt']))
 expected={'pytorch_model-00001-of-00002.bin':'ca4cdc96237832a9eb3486274de0b10fa7235d5007ee6cff49320c7250970fe3','pytorch_model-00002-of-00002.bin':'d445a50608f025a39ac75c6da13cb9d12a4cc923f6e1b9abc4b767408eac0729'}
 for name,digest in expected.items():assert sha(snapshot/name)==digest,name
 refs=args.cache_dir.resolve()/'models--q-future--one-align/refs';refs.mkdir(parents=True,exist_ok=True);(refs/'main').write_text(REV)
 save(O/'model_source.json',{'revision':REV,'verified_weight_sha256':expected})
 from pyiqa.archs.qalign_arch import QAlign,MPLUGOwl2LlamaForCausalLM
 cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.manual_seed(20261018);np.random.seed(20261018)
 oldenv=json.loads(args.fourbit_environment.read_text('utf-8'))
 versions={k:meta.version(k) for k in oldenv['versions']}
 assert versions==oldenv['versions'],(versions,oldenv['versions'])
 impl=Path(pyiqa.__file__).parent
 for p,h in oldenv['implementation_sha256'].items():assert sha(impl/p.replace('\\','/'))==h,p
 # Retain exact installed QAlign __init__/preprocess/forward and model.score functions.
 # Only augment the model loading call. Do not invoke InferenceModel's blanket .to(cuda).
 original_loader=MPLUGOwl2LlamaForCausalLM.from_pretrained
 cpu_budget=int((avail-3*2**30)//2**30)
 maxmem={0:'5GiB','cpu':f'{cpu_budget}GiB'}
 def fp16_load(cls,*args,**kwargs):
  assert args==('q-future/one-align',)
  assert 'quantization_config' not in kwargs
  kwargs.update(torch_dtype=torch.float16,device_map='auto',max_memory=maxmem,low_cpu_mem_usage=True,
                offload_folder=str(O/'offload'),offload_state_dict=True,local_files_only=True,revision=REV)
  return original_loader(*args,**kwargs)
 env={'versions':versions,'same_source_hashes_as_4bit':True,'revision':REV,'load_dtype':'float16','device_map_request':'auto',
      'max_memory':{str(k):v for k,v in maxmem.items()},'timing_is_offload_feasibility_only':True,'gpu':torch.cuda.get_device_name(),
      'free_total_cuda_before_load':torch.cuda.mem_get_info(),'worker_script_sha256':sha(__file__)}
 save(O/'environment.json',env)
 print('LOAD FP16',now(),maxmem,flush=True);t0=time.perf_counter()
 with contextlib.ExitStack() as stack:
  from unittest.mock import patch
  stack.enter_context(patch.object(MPLUGOwl2LlamaForCausalLM,'from_pretrained',classmethod(fp16_load)))
  metric=QAlign(dtype='fp16')
 metric.eval();model=metric.model;loadsec=time.perf_counter()-t0
 mapping=getattr(model,'hf_device_map',{})
 assert any(str(v)=='cpu' for v in mapping.values()),mapping
 assert not any(str(v)=='disk' for v in mapping.values()),'Disk-offloaded weights disallowed'
 assert not getattr(model,'is_loaded_in_4bit',False) and not getattr(model,'is_loaded_in_8bit',False)
 assert all(p.dtype==torch.float16 for p in model.parameters() if p.is_floating_point()),'Non-fp16 floating weights'
 assert not any(type(m).__name__ in ['Linear4bit','Linear8bitLt'] for m in model.modules())
 processor=metric.image_processor.to_dict();assert processor==oldenv['internal_processor']
 env.update(model_load_seconds=loadsec,device_map={k:str(v) for k,v in mapping.items()},internal_processor=processor,
            floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
            preferential_token_ids=model.preferential_ids_,preferential_levels=['excellent','good','fair','poor','bad'],weights=[5,4,3,2,1],
            prompt='USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is',
            free_total_cuda_after_load=torch.cuda.mem_get_info(),free_ram_after_load_bytes=memory())
 save(O/'environment.json',env)
 def tensor(path):
  t0=time.perf_counter();bgr=cv2.imread(path,cv2.IMREAD_COLOR);assert bgr is not None,path
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
  h=hashlib.sha256(rgb.tobytes()).hexdigest();x=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
  prep=(time.perf_counter()-t0)*1000;t0=time.perf_counter();x=x.to('cuda');torch.cuda.synchronize();return x,prep,(time.perf_counter()-t0)*1000,h
 data=pd.read_csv(args.manifest,dtype={'scene':str},float_precision='round_trip')
 four=pd.read_csv(args.fourbit_scores,dtype={'scene':str},float_precision='round_trip').set_index(['dataset','name'])
 square=data[data.dataset=='squarelol'].sort_values('name').reset_index(drop=True)
 pilot=square.iloc[np.random.default_rng(20261021).choice(len(square),size=50,replace=False)]
 assert len(data)==len(four) and len(pilot)==50
 dest=O/'scores.csv';rows={};fields=['dataset','name','scene','score','prepare_ms','h2d_ms','inference_ms','wall_ms','rgb256_sha256','cuda_allocated_bytes','cuda_reserved_bytes','phase','scored_utc']
 if dest.exists() and dest.stat().st_size:
  # Recover only an interrupted final physical line; preserve the original before truncation.
  raw=dest.read_bytes()
  if not raw.endswith(b'\n'):
   stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
   (O/f'scores_before_tail_recovery_{stamp}.csv').write_bytes(raw)
   cut=raw.rfind(b'\n')+1
   if cut==0:raise RuntimeError('CSV header itself is incomplete; inspect before resume')
   dest.write_bytes(raw[:cut])
   save(O/f'tail_recovery_{stamp}.json',{'dropped_incomplete_bytes':len(raw)-cut,'reason':'interruption during final CSV append; image will be recomputed'})
  with dest.open('r',encoding='utf-8',newline='') as f:
   for r in csv.DictReader(f):
    key=(r['dataset'],r['name']);assert key not in rows,key
    assert np.isfinite(float(r['score'])) and r['rgb256_sha256']==four.loc[key,'rgb256_sha256']
    assert all(r[c] is not None and r[c]!='' for c in fields),'Incomplete CSV row; stop for exact repair'
    rows[key]=r
 with dest.open('a',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields)
  if not rows:writer.writeheader();f.flush();os.fsync(f.fileno())
  def evaluate(row,phase):
   key=(row.dataset,row.name)
   if key in rows:return
   state(status=phase,completed=len(rows),current_dataset=row.dataset,current_image=row.name,current_started_utc=now())
   wall=time.perf_counter();x,prep,h2d,h=tensor(row.path)
   assert h==four.loc[key,'rgb256_sha256'],('RGB differs',key)
   assert x.shape==(1,3,256,256) and x.min()>=0 and x.max()<=1
   torch.cuda.synchronize();t0=time.perf_counter()
   v=float(metric(x,task_='quality',input_='image').item());torch.cuda.synchronize();ms=(time.perf_counter()-t0)*1000
   wallms=(time.perf_counter()-wall)*1000
   if not np.isfinite(v):
    state(status='stopped_nonfinite',failed_dataset=row.dataset,failed_image=row.name,invalid_score=str(v));raise FloatingPointError('Nonfinite fp16 score; stopping')
   rec=dict(dataset=row.dataset,name=row.name,scene=row.scene,score=v,prepare_ms=prep,h2d_ms=h2d,inference_ms=ms,wall_ms=wallms,
       rgb256_sha256=h,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),phase=phase,scored_utc=now())
   writer.writerow(rec);f.flush();os.fsync(f.fileno());rows[key]=rec
   state(status=phase,completed=len(rows),last_completed_dataset=row.dataset,last_completed_image=row.name,last_wall_seconds=wallms/1000)
   print(f'{now()} {phase} {len(rows)} {row.dataset}/{row.name} {wallms/1000:.2f}s score={v}',flush=True)
  with torch.inference_mode():
   for row in pilot.itertuples(index=False):evaluate(row,'pilot')
   a=np.array([float(rows[(r.dataset,r.name)]['score']) for r in pilot.itertuples()]);b=np.array([four.loc[(r.dataset,r.name),'score'] for r in pilot.itertuples()])
   seconds=np.array([float(rows[(r.dataset,r.name)]['wall_ms'])/1000 for r in pilot.itertuples()])
   total_hours=(seconds.mean()*len(data)+loadsec)/3600
   selected=['squarelol','rlie'] if total_hours>14 else ['squarelol','rlie','lieq']+[ds for ds in data.dataset.unique() if ds not in ['squarelol','rlie','lieq']]
   selected=[s for s in selected if s in set(data.dataset)]
   selected_n=int(data.dataset.isin(selected).sum())
   if not (O/'pilot.json').exists():
    save(O/'pilot.json',{'n':50,'seed':20261021,'pearson_fp16_4bit':float(pearsonr(a,b).statistic),'srcc_fp16_4bit':float(spearmanr(a,b).statistic),
      'nan_count':int(np.isnan(a).sum()),'inf_count':int(np.isinf(a).sum()),'mean_wall_seconds':float(seconds.mean()),'median_wall_seconds':float(np.median(seconds)),
      'p95_wall_seconds':float(np.quantile(seconds,.95)),'estimated_all_hours':float(total_hours),'selected_datasets':selected,'selected_images':selected_n,
      'estimated_selected_hours':float((seconds.mean()*selected_n+loadsec)/3600),'estimated_remaining_hours':float(seconds.mean()*(selected_n-len(rows))/3600),
      'omitted_datasets':[s for s in data.dataset.unique() if s not in selected],'model_load_seconds':loadsec,'pilot_includes_first_call':True,'created_utc':now()})
   decision=json.loads((O/'pilot.json').read_text('utf-8'));selected=decision['selected_datasets'];selected_n=decision['selected_images']
   state(status='scoring',pilot=decision,selected_datasets=selected,total=selected_n)
   print('PILOT',json.dumps(decision),flush=True)
   for ds in selected:
    for row in data[data.dataset==ds].itertuples(index=False):evaluate(row,'scoring')
 state(status='scores_complete',completed=len(rows),finished_scoring_utc=now(),scores_sha256=sha(dest))
 del metric,model;import gc;gc.collect();torch.cuda.empty_cache()
 state(status='complete',finished_utc=now())
 print('COMPLETE. Run analyze.py separately; fp16 offload times are not a cost benchmark.',flush=True)

if __name__=='__main__':
 try:worker()
 except Exception:
  error=traceback.format_exc();print(error,flush=True)
  status=json.loads(STATE.read_text('utf-8')).get('status') if STATE.exists() else ''
  if status!='stopped_nonfinite':state(status='error',error=error,finished_utc=now())
  raise