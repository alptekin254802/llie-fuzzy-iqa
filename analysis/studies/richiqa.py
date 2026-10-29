"""Pinned RichIQA runner; official source is imported and never edited."""
from pathlib import Path
import os,sys,json,csv,time,hashlib,datetime,traceback,argparse
from analysis.studies.config import OUT,REPO,location
O=OUT;D=OUT/'richiqa_release';DS=['lol','squarelol','lieq','rlie'];COUNTS={'lol':(2425,485),'squarelol':(2900,290),'lieq':(1000,100),'rlie':(1540,154)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def state(**kwargs):
 p=O/'richiqa_run.json';d=json.loads(p.read_text());d.update(kwargs);d['updated_utc']=now();save(p,d)
def setup():
 import requests,zipfile,shutil
 github='bfcf34939937e8fd2a55179982d648c4dfb29c37';hf='af8ff09037533c265c8a57d981319b549b253a4b';D.mkdir(parents=True,exist_ok=True)
 urls={'source.zip':f'https://codeload.github.com/YixuanGao98/RichIQA/zip/{github}','best.pkl':f'https://huggingface.co/GYX98/RichIQA/resolve/{hf}/best.pkl?download=true'}
 expected={'source.zip':'22cf818dbb0e68f5ef22035c07847314aea936679279e7539756c2229bc96fd4','best.pkl':'22fda9c41ef3b03f71c828565056dd92993a30a299b93aab120ce4533f5ebdb7','CvT-21-384x384-IN-22k.pth':'e630d7b77ac3777832a1d4c89e989dd2b51459d6eebe8f96816e4d794f51a14f'}
 for name,url in urls.items():
  p=D/name
  if not p.exists():
   with requests.get(url,stream=True,timeout=(20,90)) as r:
    r.raise_for_status()
    with p.open('wb') as f:
     for block in r.iter_content(1024*1024):f.write(block)
  assert sha(p)==expected[name],name
 S=D/f'RichIQA-{github}'
 if not S.exists():
  with zipfile.ZipFile(D/'source.zip') as z:
   for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
   z.extractall(D)
 backbone=location('richiqa_backbone','third_party/richiqa/CvT-21-384x384-IN-22k.pth');assert backbone.exists(),'Download the exact file from the authors\' OneDrive model zoo and set richiqa_backbone.'
 assert sha(backbone)==expected[backbone.name];target=S/'checkpoint'/backbone.name
 if target.resolve()!=backbone.resolve():shutil.copy2(backbone,target)
 files={name:{'path':str(D/name),'sha256':expected[name],'url':url} for name,url in urls.items()}
 files[backbone.name]={'path':str(target),'sha256':expected[backbone.name],'url':'https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al'}
 save(O/'richiqa_sources.json',{'github_commit':github,'hf_commit':hf,'source_directory':str(S),'source_hashes':{str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in ['.py','.md']},'files':files})
 print('Pinned source and both checkpoints verified.')
def prepare():
 import pandas as pd,numpy as np,cv2
 from analysis.studies import extra_metrics,qalign,common as c
 extra_metrics.prepare();qalign.prepare()
 old=pd.read_csv(O/'metric_manifest.csv',dtype={'scene':str},float_precision='round_trip');q=pd.read_csv(O/'qalign_manifest.csv',dtype={'scene':str},float_precision='round_trip');oof=pd.read_csv(REPO/'results/nr_fuzzy_oof.csv',float_precision='round_trip').rename(columns={'prediction':'oof_nr_free'});oof=oof.merge(c.lol_table()[['filename','method','lpips']],on=['filename','method'],validate='one_to_one');oof['name']=oof.filename+'__'+oof.method;rows=[]
 baseline=pd.read_csv(REPO/'results/studies/richiqa_input_hashes.csv',dtype={'scene':str})
 for ds in DS:
  x=old[old.dataset==ds].copy().sort_values('name');assert (len(x),x.scene.nunique())==COUNTS[ds]
  if ds=='lol':
   x=x.merge(oof[['name','fold','oof_nr_free','lpips']],on='name',validate='one_to_one');assert np.allclose(x.target,-x.lpips,rtol=0,atol=1e-12);x['nr_free']=x.oof_nr_free
  else:
   x=x.merge(q[q.dataset==ds][['name','mos','nr_free']],on='name',validate='one_to_one');assert np.allclose(x.target,x.mos,rtol=0,atol=1e-12);x['target']=x.mos;x['fold']=-1
  lookup=baseline[baseline.dataset==ds].set_index('name').rgb256_sha256.to_dict()
  for r in x.itertuples():
   bgr=cv2.imread(r.path);assert bgr is not None and bgr.shape==(256,256,3),r.path
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h=hashlib.sha256(rgb.tobytes()).hexdigest();assert np.array_equal(rgb,cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA));assert h==lookup[r.name],(ds,r.name)
   rows.append({'dataset':ds,'name':r.name,'scene':r.scene,'target':float(r.target),'nr_free':float(r.nr_free),'fold':int(r.fold),'path':r.path,'rgb256_sha256':h})
 pd.DataFrame(rows).to_csv(O/'richiqa_manifest.csv',index=False)
 (O/'richiqa_PLAN.md').write_bytes((REPO/'docs/RICHIQA.md').read_bytes())
 save(O/'richiqa_inputs.json',{'manifest_sha256':sha(O/'richiqa_manifest.csv'),'images':len(rows),'counts':COUNTS,'all_pixels_match_recorded_inputs':True})
 if not (O/'richiqa_run.json').exists():save(O/'richiqa_run.json',{'status':'prepared','started_utc':now(),'completed':0,'plan_sha256':sha(O/'richiqa_PLAN.md')})
 print('Prepared',len(rows),'verified inputs',flush=True)
def run():
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
 import torch,cv2,numpy as np,pandas as pd,importlib.metadata as meta,platform,subprocess
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(20261019);np.random.seed(20261019)
 source=json.loads((O/'richiqa_sources.json').read_text());S=Path(source['source_directory'])
 def check_source():
  for rel,h in source['source_hashes'].items():assert sha(S/rel)==h,rel
  for name in ['best.pkl','CvT-21-384x384-IN-22k.pth']:assert sha(source['files'][name]['path'])==source['files'][name]['sha256']
 check_source();sys.path.insert(0,str(S));os.chdir(S)
 import inference as official
 state(status='loading',worker_pid=os.getpid());print('Loading official inference.py',now(),flush=True)
 model=official.load_model(str(D/'best.pkl'))
 assert not model.training and all(p.device.type=='cuda' for p in model.parameters())
 ref=torch.load(D/'best.pkl',map_location='cpu',weights_only=True);assert set(ref)==set(model.state_dict());assert all(torch.equal(v.cpu(),ref[k]) for k,v in model.state_dict().items());del ref
 env={'python':sys.version,'versions':{k:meta.version(k) for k in ['torch','torchvision','timm','einops','numpy','scipy','pandas','opencv-python','pillow']},'gpu':torch.cuda.get_device_name(),'cpu':platform.processor(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'batch_size':1,'dtype':'float32','official_transform':str(official.transform),'inference_source_sha256':sha(S/'inference.py'),'github_commit':source['github_commit'],'hf_commit':source['hf_commit'],'strict_full_checkpoint_equality':True,'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu','--format=csv']).decode(),'timing_boundary':'Complete unchanged predict_image_quality call on prepared AREA256-equivalent PNG: decode, author preprocessing, transfer, model, scalar; excludes model loading and external AREA preparation.'}
 save(O/'richiqa_environment.json',env)
 data=pd.read_csv(O/'richiqa_manifest.csv',dtype={'scene':str},float_precision='round_trip');warm=data.groupby('dataset',sort=False).head(3)
 def infer(net,r,device):
  if device=='cuda':torch.cuda.synchronize()
  t=time.perf_counter();dist,score=official.predict_image_quality(net,r.path)
  if device=='cuda':torch.cuda.synchronize()
  ms=(time.perf_counter()-t)*1000
  assert dist.shape==(1,5) and np.isfinite(dist).all() and abs(float(dist.sum())-1)<1e-5 and 1<=score<=5,(r.name,dist,score)
  return float(score),ms,dist[0].tolist()
 pilot=[]
 for r in warm.itertuples():
  v,ms,dist=infer(model,r,'cuda');pilot.append({'dataset':r.dataset,'name':r.name,'score':v,'ms':ms})
  if len(pilot)==1:state(status='setup_succeeded',first_valid_score_utc=now())
 save(O/'richiqa_pilot.json',{'calls':pilot,'excluded_from_timing':True})
 cpu_path=O/'richiqa_cpu.csv'
 if not cpu_path.exists():
  state(status='cpu_timing');net=model.module.cpu();torch.cuda.empty_cache();selected=data.groupby('dataset',sort=False).head(25);rows=[]
  for session in range(1,4):
   for r in warm.itertuples():infer(net,r,'cpu')
   for r in selected.itertuples():
    v,ms,dist=infer(net,r,'cpu');rows.append({'session':session,'dataset':r.dataset,'name':r.name,'score':v,'ms':ms})
   pd.DataFrame(rows).to_csv(cpu_path,index=False);state(status='cpu_timing',cpu_calls=len(rows));print('CPU session',session,'complete',flush=True)
  model.cuda()
 else:
  assert len(pd.read_csv(cpu_path))==300,'Incomplete CPU benchmark: diagnose before resume'
 # Warm GPU again after the CPU stage; these repeated warm-ups are never timed observations.
 for r in warm.itertuples():infer(model,r,'cuda')
 dest=O/'richiqa_scores.csv';done=set()
 if dest.exists():
  prior=pd.read_csv(dest);assert not prior.duplicated(['dataset','name']).any();done=set(zip(prior.dataset,prior.name))
 fields=['dataset','name','scene','score','ms','rgb256_sha256','p1','p2','p3','p4','p5']
 with dest.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if not done:w.writeheader();f.flush()
  for r in data.itertuples():
   if (r.dataset,r.name) in done:continue
   v,ms,dist=infer(model,r,'cuda');w.writerow({'dataset':r.dataset,'name':r.name,'scene':r.scene,'score':v,'ms':ms,'rgb256_sha256':r.rgb256_sha256,**{f'p{k+1}':v for k,v in enumerate(dist)}});f.flush();done.add((r.dataset,r.name))
   if len(done)%100==0:state(status='gpu_scoring',completed=len(done),total=len(data),last_ms=ms);print(now(),len(done),'/',len(data),round(ms,2),'ms',flush=True)
 check_source();env['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();save(O/'richiqa_environment.json',env);state(status='completed',completed=len(done),total=len(data),finished_utc=now(),raw_sha256=sha(dest));print('COMPLETE',flush=True)
if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='setup':setup()
 elif stage=='prepare':prepare()
 elif stage=='run':run()
 elif stage=='analyze':
  from analysis.studies.richiqa_summary import main
  main()
 else:raise SystemExit('Expected setup, prepare, run or analyze')
