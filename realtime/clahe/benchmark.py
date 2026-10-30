"""Repeat the original and exact optimised CLAHE timing in fresh processes."""
from .core import *
from . import fast as fast_clahe
from analysis.parameter_selection import candidates as op, family_ops as fast
from analysis.parameter_selection.scorers import load_models
from analysis.parameter_selection.common import configure,output_dir
from types import SimpleNamespace
import argparse
read_plan_families=json.loads((DATA/'screen/PLAN.json').read_text())['families']
SHA=sha;NOW=now;csv=writecsv

def predict_model(models,selector,frame):
    model=models[selector]
    if selector in ('RF3','RF4'):frame=frame[list(model.feature_names_in_)]
    return model.predict(frame)

def choose(scores,params,default):
    scores=np.asarray(scores,float);params=np.asarray(params,float)
    mask=scores>=scores.max()-1e-9;distance=np.round(abs(params-default),12)
    mask &= distance==np.where(mask,distance,np.inf).min()
    return int(np.where(mask,params,np.inf).argmin())

def invoke(raw,family,selector,models,controls):
    low=op.exposed(raw,1.)
    default=float(read_plan_families[family]['default'])
    if selector in ('default','training_fixed'):
        value=default if selector=='default' else controls[family]['fixed_param']
        return float(value),fast.single_candidate(low,family,value)
    candidates=fast.family_candidates(low,family);params=np.array([p for p,_ in candidates])
    names=('brightness',) if selector=='brightness_tuned' else (('entropy','contrast','sharpness','brightness') if selector in ('BrightnessCorrection_lam01','RF4') else ('entropy','contrast','sharpness'))
    frame=pd.DataFrame([fast.required_features(im,names) for _,im in candidates])
    scores=-np.abs(frame.brightness.to_numpy()-controls[family]['brightness_target']) if selector=='brightness_tuned' else predict_model(models,selector,frame)
    index=choose(scores,params,default)
    return float(params[index]),candidates[index][1]

old=SimpleNamespace(invoke=invoke)
def setup():
 models=load_models('final');controls=json.loads((DATA/'controls.json').read_text())['final']['families']
 return models,controls,fast_clahe.Engine(models=models)
def load_inputs():
 sample=read(DATA/'timing_inputs.csv').fillna('');raw=[];native=[];sized=[];records=[]
 for row in sample.to_dict('records'):
  raw.append(read_rgb(row));path=resolve(row['native_input_path']);b=cv2.imread(str(path),cv2.IMREAD_COLOR);assert b.shape==(400,600,3)
  native.append(b);sized.append(cv2.resize(b,(640,480),interpolation=cv2.INTER_LINEAR))
  records.append({'scene':row['scene'],'native_sha256':sha(path),'bgr600x400_sha256':__import__('hashlib').sha256(b.tobytes()).hexdigest(),'bgr640x480_sha256':__import__('hashlib').sha256(sized[-1].tobytes()).hexdigest()})
 return sample,{'256':raw,'600x400':native,'640x480':sized},records
def check():return {'release_sha256':check_freeze()}
def prepare():
 sample,images,records=load_inputs();models,controls,engine=setup();checked=0
 for size in ['600x400','640x480']:
  for bgr in images[size]:
   x=prep(bgr);a,im=invoke(x,'clahe','NR-mono',models,controls);v,out=engine.search(x,'NR-mono')
   assert a==v and im.tobytes()==out.tobytes();checked+=1
 save(R/'timing/INPUTS.json',{'records':records,'all_exact':True,'additional_boundary_choices_checked':checked})

def session(boundary,number):
 check();assert boundary in ['primary','preprocessing'] and number in [1,2,3]

 out=R/'timing'/f'{boundary}_session{number}.csv';assert not out.exists()
 if number>1:assert (R/'timing'/f'{boundary}_session{number-1}.json').exists()
 sample,inputs,records=load_inputs();frozen=json.loads((R/'timing/INPUTS.json').read_text());assert records==frozen['records']
 models,controls,engine=setup()
 if boundary=='primary':cases=[(v,s,'256') for v in ['old','fast'] for s in ['NR-free','NR-mono','Linear3','training_fixed']]
 else:cases=[(v,'NR-mono',size) for v in ['old','fast'] for size in ['600x400','640x480']]
 def call(image,case):
  v,name,size=case;x=image if size=='256' else prep(image)
  return old.invoke(x,'clahe',name,models,controls) if v=='old' else engine.search(x,name)
 for case in cases:
  for i in range(20):call(inputs[case[2]][i],case)
 cpu_before=psutil.cpu_percent(interval=5);
 if cpu_before>10:raise RuntimeError('Background load exceeds 10 percent; no timed calls recorded')
 start=NOW();t0=time.perf_counter();rng=np.random.default_rng((20261030 if boundary=='primary' else 20261031)+number);rows=[];checksum=0.
 for repeat in range(3):
  for i in rng.permutation(100):
   for j in rng.permutation(len(cases)):
    case=cases[j];image=inputs[case[2]][int(i)];beg=time.perf_counter_ns();param,imageout=call(image,case);elapsed=(time.perf_counter_ns()-beg)/1e6;checksum+=param+float(imageout[0,0,0]);rows.append({'boundary':boundary,'session':number,'pid':os.getpid(),'repeat':repeat,'scene':sample.scene.iloc[i],'version':case[0],'selector':case[1],'input_size':case[2],'param':param,'milliseconds':elapsed})
 end=NOW();cpu_after=psutil.cpu_percent(interval=.2);csv(out,rows)
 save(out.with_suffix('.json'),{'boundary':boundary,'session':number,'pid':os.getpid(),'start_utc':start,'end_utc':end,'elapsed_seconds':time.perf_counter()-t0,'calls':len(rows),'checksum':checksum,'raw_sha256':SHA(out),'code_freeze_sha256':check_freeze(),'system':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads(),'threadpools':threadpool_info(),'cpu_before':cpu_before,'cpu_after':cpu_after,'available_ram_bytes':psutil.virtual_memory().available}});print('SESSION_COMPLETE',boundary,number,len(rows),flush=True)

def aggregate(boundary):
 check();raw=[];meta=[]
 for n in [1,2,3]:
  p=R/'timing'/f'{boundary}_session{n}.csv';j=json.loads(p.with_suffix('.json').read_text());assert j['raw_sha256']==SHA(p);raw.append(pd.read_csv(p,dtype={'scene':str,'input_size':str}));meta.append(j)
 assert len({j['pid'] for j in meta})==3 and all(meta[i]['end_utc']<meta[i+1]['start_utc'] for i in range(2));allraw=pd.concat(raw,ignore_index=True);summary=[];ratios=[];budget=1000/30
 for name,table in [(str(i+1),v) for i,v in enumerate(raw)]+[('pooled',allraw)]:
  for keys,z in table.groupby(['version','selector','input_size']):
   v=z.milliseconds;summary.append({'session':name,'version':keys[0],'selector':keys[1],'input_size':keys[2],'calls':len(z),'median_ms':float(v.median()),'p95_ms':float(v.quantile(.95)),'p99_ms':float(v.quantile(.99)),'max_ms':float(v.max()),'deadline_misses':int((v>budget).sum()),'deadline_miss_fraction':float((v>budget).mean())})
  keys=['session','repeat','scene','selector','input_size'];wide=table.pivot(index=keys,columns='version',values='milliseconds');params=table.pivot(index=keys,columns='version',values='param');assert (params.old==params.fast).all()
  for (selector,size),z in wide.groupby(level=['selector','input_size']):ratios.append({'session':name,'selector':selector,'input_size':size,'paired_median_speedup':float((z.old/z.fast).median()),'median_ratio':float(z.old.median()/z.fast.median())})
 summary=pd.DataFrame(summary);csv(R/'timing'/f'{boundary}_summary.csv',summary);csv(R/'timing'/f'{boundary}_ratios.csv',ratios)
 target=summary[(summary.version=='fast')&(summary.selector=='NR-mono')&(summary.session!='pooled')]
 assert len(target)==(3 if boundary=='primary' else 6)
 result={'status':'COMPLETE','utc':NOW(),'boundary':boundary,'calls':len(allraw),'separate_sequential_processes':True,'nr_mono_each_session_p95_under_budget':bool((target.p95_ms<budget).all()),'budget_ms':budget,'summary_sha256':SHA(R/'timing'/f'{boundary}_summary.csv'),'raw_sessions':[{k:j[k] for k in ['session','pid','start_utc','end_utc','raw_sha256']} for j in meta],'scope':'Resident RGB256 to output, includingexposure' if boundary=='primary' else 'Resident nativeBGR8 pluscolorconversion,uint8bilinearresize,float32normalisation and completeCLAHEsearch; excludes capture/decode/display','not_a_worstcase_or_live_camera_guarantee':True}
 save(R/'timing'/f'{boundary}_RESULT.json',result);print(json.dumps(result),flush=True);print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','session','aggregate']);p.add_argument('--boundary',choices=['primary','preprocessing'],default='primary');p.add_argument('--number',type=int,choices=[1,2,3]);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 configure(a.config);R=output_dir(a.output);(R/'timing').mkdir(exist_ok=True)
 if a.stage=='prepare':prepare()
 elif a.stage=='session':session(a.boundary,a.number)
 else:aggregate(a.boundary)
