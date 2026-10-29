"""Independent reader: does not import capture.py, analyze.py or optimized_loop.py."""
import os,sys,json,hashlib,argparse,importlib.metadata
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT.parent))
import cv2,numpy as np,pandas as pd,torch,pyiqa
from reference import basic,rgb8
from reference import predict
from preprocessing.enhancement.methods import apply_gamma_correction
from evaluation.metrics import compute_psnr,compute_ssim
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def independently_select(group,method,previous,fixed):
 g=group[group.adaptive_candidate].copy()
 if method=='default':return float(group.loc[np.isclose(group.gamma,2/3,rtol=0,atol=1e-12),'gamma'].iloc[0])
 if method=='fixed':return float(group.loc[np.isclose(group.gamma,fixed,rtol=0,atol=1e-12),'gamma'].iloc[0])
 if method=='brightness':scores=-(g.brightness-.45).abs().to_numpy()
 elif method=='oracle':scores=-g.lpips.to_numpy()
 else:scores=g[method].to_numpy()*(-1 if method=='niqe' else 1)
 finite=np.isfinite(scores)
 indices=np.flatnonzero(finite&np.isclose(scores,np.max(scores[finite]),rtol=0,atol=1e-9)) if finite.any() else np.arange(len(g))
 # Independent Python ordering instead of production np.lexsort.
 choices=g.iloc[indices].gamma.tolist()
 return min(choices,key=lambda v:(float(np.round(abs(v-previous),12)),float(np.round(abs(v-2/3),12)),v))
def exact_top1_and_pairs_only(pred,target):
 # Same exact-tie rules, independent loops; no correlation test or p-value.
 best=max(pred);top=[i for i,value in enumerate(pred) if value==best]
 top1=sum(target[i]==max(target) for i in top)/len(top)
 correct=0.0;pairs=0
 for i in range(len(pred)):
  for j in range(i+1,len(pred)):
   if target[i]==target[j]:continue
   pairs+=1
   if pred[i]==pred[j]:correct+=.5
   elif (pred[i]>pred[j])==(target[i]>target[j]):correct+=1
 return {'top1':top1,'pair_acc':correct/max(pairs,1),'pairs':pairs}

def assert_camera(info,plan,exposure):
 assert info['fourcc_text']=='YUY2' and info['fourcc_numeric']==844715353 and info['backend']=='DSHOW' and info['width']==640 and info['height']==480
 for key,value in plan['capture']['controls'].items():assert info['controls'][key]==(-1 if key=='AUTO_EXPOSURE' else 2 if key=='AUTOFOCUS' else value)
 if exposure is not None:assert info['controls']['EXPOSURE']==exposure

def independent_description(selected,result):
 scenes=sorted(selected.scene.unique());n=len(scenes);assert n==result['n_scenes']
 assert result['interpretation']=='descriptive only' and result['bootstrap_replicates']==2000 and result['seed']==20261018
 draw=np.random.default_rng(20261018).integers(0,n,size=(2000,n)) if n>=2 else None
 values={}
 for method,metrics in result['scene_mean_metrics_by_selector'].items():
  values[method]={}
  for metric,record in metrics.items():
   array=np.array([selected[(selected.scene==s)&(selected.selector==method)][metric].mean() for s in scenes]);values[method][metric]=array
   assert abs(array.mean()-record['mean'])<1e-12
   if draw is None:assert record['descriptive_bootstrap_interval95'] is None
   else:assert np.allclose(np.quantile(array[draw].mean(axis=1),[.025,.975]),record['descriptive_bootstrap_interval95'],atol=1e-12,rtol=0)
 diff=values['fuzzy']['lpips']-values['fixed']['lpips'];record=result['scene_mean_lpips_difference_fuzzy_minus_fixed']
 assert abs(diff.mean()-record['mean'])<1e-12
 if draw is None:assert record['descriptive_bootstrap_interval95'] is None
 else:assert np.allclose(np.quantile(diff[draw].mean(axis=1),[.025,.975]),record['descriptive_bootstrap_interval95'],atol=1e-12,rtol=0)
 diff_default=values['fuzzy']['lpips']-values['default']['lpips'];record=result['scene_mean_lpips_difference_fuzzy_minus_default']
 assert abs(diff_default.mean()-record['mean'])<1e-12
 if draw is None:assert record['descriptive_bootstrap_interval95'] is None
 else:assert np.allclose(np.quantile(diff_default[draw].mean(axis=1),[.025,.975]),record['descriptive_bootstrap_interval95'],atol=1e-12,rtol=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--scene',default='dry_run_3');p.add_argument('--run');a=p.parse_args();plan=js(ROOT/'PLAN.json');assert sha(ROOT/'PLAN.json')==(ROOT/'PLAN.sha256').read_text().split()[0]
 amendment=js(ROOT/'PLAN_AMENDMENT_1.json');assert sha(ROOT/'PLAN_AMENDMENT_1.json')==(ROOT/'PLAN_AMENDMENT_1.sha256').read_text().split()[0]
 amended=js(ROOT/'PLAN_AMENDMENT_3.json');assert sha(ROOT/'PLAN_AMENDMENT_3.json')==(ROOT/'PLAN_AMENDMENT_3.sha256').read_text().split()[0]
 assert amended['framework']['kind']=='exploratory' and amended['framework']['p_values'] is False
 run=a.run or ('smoke_'+a.scene if a.smoke else None);assert run and Path(run).name==run,'Use --run RUN_NAME'
 dest=ROOT/'derived'/run;data=pd.read_csv(dest/'candidates.csv',float_precision='round_trip',converters={'room_label':str,'direction_label':str});chosen=pd.read_csv(dest/'selections.csv',float_precision='round_trip')
 assert data.dry_run.eq(a.smoke).all() and chosen.dry_run.eq(a.smoke).all()
 scope_path=ROOT/'analysis_runs'/run/'INPUTS.json'
 if scope_path.exists():
  scope=js(scope_path)
  assert sorted(data.scene.unique())==scope['scene_ids']==sorted(chosen.scene.unique())
  assert sorted(data.room_label.unique())==scope['rooms']==sorted(chosen.room_label.unique())
  assert sha(ROOT/'EXCLUSIONS.json')==scope['exclusions_sha256']
  assert not set(scope['excluded_scene_ids']) & set(data.scene)
  for scene in scope['scene_ids']:assert sha(ROOT/'data/accepted'/scene/'metadata.json')==scope['scene_metadata_sha256'][scene]
 assert importlib.metadata.version('pyiqa')==plan['scoring']['pyiqa']
 mods={key:js(REPO/plan['models'][name]['path']) for key,name in [('fuzzy','nr_free'),('nr_mono','nr_mono')]}
 for s in plan['models'].values():assert sha(REPO/s['path'])==s['sha256']
 device='cuda' if torch.cuda.is_available() else 'cpu';metrics={m:pyiqa.create_metric(m,device=device) for m in ['lpips','clipiqa+','niqe']}
 def ten(v):return torch.from_numpy(np.ascontiguousarray(v)).permute(2,0,1)[None].to(device)
 count=0;maxdiff={m:0. for m in ['lpips','clipiqa+','niqe','psnr','ssim']};verified_choices=0;qc_records=[]
 with torch.inference_mode():
  for scene,x in data.groupby('scene',sort=True):
   folder=ROOT/'data'/('dry_run' if a.smoke else 'accepted')/scene;meta=js(folder/'metadata.json');assert meta['dry_run']==a.smoke and meta['qc']['passed']
   assert meta['amendment_sha256']==sha(ROOT/'PLAN_AMENDMENT_1.json') and meta['E0']-4>=-7
   assert meta['amendment_3_sha256']==sha(ROOT/'PLAN_AMENDMENT_3.json') and meta['analysis_framework']=='exploratory'
   assert meta['excluded_from_all_analyses']==a.smoke and meta['room_label'].strip() and meta['direction_label'].strip()
   assert x.room_label.eq(meta['room_label']).all() and x.direction_label.eq(meta['direction_label']).all()
   assert_camera(meta['before_scene_readback'],plan,None);assert_camera(meta['final_camera_readback'],plan,meta['E0'])
   assert len(meta['frames'])==20 and sorted((r['stop'],r['repetition']) for r in meta['frames'])==[(s,j) for s in [1,2,3,4] for j in range(5)]
   refs=[];native=[]
   for side in ['reference_begin','reference_end']:
    r=meta[side];assert len(r['raw_frames'])==16;total=np.zeros((480,640,3),dtype=np.uint16)
    assert_camera(r['camera_readback'],plan,meta['E0'])
    for rr in r['raw_frames']:
     assert_camera(rr['camera_readback'],plan,meta['E0'])
     assert sha(folder/rr['file'])==rr['sha256'];total+=cv2.imread(str(folder/rr['file'])).astype(np.uint16)
    assert sha(folder/r['file'])==r['sha256'];stored=cv2.imread(str(folder/r['file']),cv2.IMREAD_UNCHANGED);assert np.array_equal(total,stored)
    native.append(stored.astype(np.float64)/4080)
    avg=cv2.cvtColor(stored,cv2.COLOR_BGR2RGB).astype(np.float32)/16;refs.append(cv2.resize(avg,(256,256),interpolation=cv2.INTER_LINEAR)/255.0)
    lum=float(cv2.cvtColor((stored.astype(np.float32)/16),cv2.COLOR_BGR2GRAY).mean()/255);clip=float(np.any(stored==4080,axis=2).mean());assert .35<=lum<=.55 and clip<.01
   mse=float(np.mean((native[0]-native[1])**2));qpsnr=-10*np.log10(mse) if mse else np.inf;assert qpsnr>=plan['QC']['begin_end_reference_psnr_db_min'];qc_records.append({'scene':scene,'qc_pass':True,'psnr':float(qpsnr) if np.isfinite(qpsnr) else None})
   raw_lum=lambda rec:float(cv2.cvtColor(cv2.imread(str(folder/rec['file'])),cv2.COLOR_BGR2GRAY).mean()/255)
   previous=float(np.mean([raw_lum(r) for r in meta['reference_begin']['raw_frames']]))
   for stop in [1,2,3,4]:
    level=float(np.mean([raw_lum(r) for r in meta['frames'] if r['stop']==stop]));assert level<=.9*previous;previous=level
   assert previous>=.084
   reference=(refs[0]+refs[1])/2;rt=ten(reference)
   for fr,z in x.groupby('frame',sort=True):
    assert len(z)==10;rec=next(r for r in meta['frames'] if r['order']==fr);assert sha(folder/rec['file'])==rec['sha256']
    rgb=cv2.cvtColor(cv2.imread(str(folder/rec['file'])),cv2.COLOR_BGR2RGB);inp=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    assert_camera(rec['camera_readback'],plan,meta['E0']-rec['stop'])
    cachefile=ROOT/'data/candidate_cache'/run/scene/f'frame_{int(fr):02d}.npz'
    assert z.input_sha256.eq(rec['sha256']).all() and z.candidate_cache_sha256.eq(sha(cachefile)).all()
    cache=np.load(cachefile)
    for row in z.itertuples():
     # closed_loop iterates NumPy GAMMAS, then a Python-float default 2/3.
     # NumPy 2 scalar promotion distinguishes those types; CSV loses the distinction.
     g=np.asarray(plan['gammas'],dtype=np.float64)[row.candidate_index] if row.candidate_index<9 else 2/3
     im=np.clip(apply_gamma_correction(inp,gamma=1/g),0,1).astype(np.float32);assert im.tobytes()==cache['images'][row.candidate_index].tobytes()
     features=basic(im);df=pd.DataFrame([features])
     for key,value in features.items():assert float(value).hex()==float(getattr(row,key)).hex(),(scene,fr,key)
     for key in ['fuzzy','nr_mono']:assert float(predict(mods[key],df,key=='nr_mono')[0]).hex()==float(getattr(row,key)).hex(),(scene,fr,key)
     assert float(cv2.cvtColor(rgb8(im),cv2.COLOR_RGB2GRAY).mean()/255).hex()==float(row.brightness).hex()
     it=ten(im);values={'lpips':float(metrics['lpips'](it,rt).item()),'clipiqa+':float(metrics['clipiqa+'](it).item()),'niqe':float(metrics['niqe'](it).item()),'psnr':float(compute_psnr(im,reference)),'ssim':float(compute_ssim(im,reference))}
     source=z.loc[z.candidate_index==row.candidate_index].iloc[0]
     for m,v in values.items():
      error=abs(v-source[m]);maxdiff[m]=max(maxdiff[m],float(error));assert np.isclose(v,source[m],atol=1e-6,rtol=1e-6),(scene,fr,m,error)
     count+=1
    print('verified frame',scene,int(fr)+1,flush=True)
   for method,ss in chosen[chosen.scene==scene].groupby('selector'):
    previous=2/3
    for row in ss.sort_values('frame').itertuples():
     actual=independently_select(x[x.frame==row.frame],method,previous,plan['fixed_gamma']['value']);assert actual==row.gamma;previous=actual;verified_choices+=1
 diagnostics=pd.read_csv(dest/'exact_top1_diagnostics.csv',float_precision='round_trip',converters={'room_label':str,'direction_label':str});diagnostics_verified=0
 for row in diagnostics.itertuples():
  a1=data[(data.scene==row.scene)&(data.frame==row.frame)&data.adaptive_candidate];target=-a1.lpips.to_numpy();method=row.selector
  pred=-np.abs(a1.brightness.to_numpy()-.45) if method=='brightness' else target if method=='oracle' else a1[method].to_numpy()*(-1 if method=='niqe' else 1)
  if not np.isfinite(pred).all() or len(np.unique(target))<2:
   assert pd.isna(row.top1) and pd.isna(row.pair_acc) and row.pairs==0
   diagnostics_verified+=1;continue
  expected=exact_top1_and_pairs_only(pred,target)
  assert np.isclose(expected['top1'],row.top1,rtol=0,atol=1e-12) and np.isclose(expected['pair_acc'],row.pair_acc,rtol=0,atol=1e-12) and expected['pairs']==row.pairs
  diagnostics_verified+=1
 independent_bootstrap=None
 if not a.smoke:
  result=js(dest/'results.json');assert result['analysis_framework']=='exploratory' and result['hypothesis_test_performed'] is False and result['p_values_computed'] is False and result['preregistered_primary_hypothesis'] is False
  assert len(chosen.scene.unique())>=20
  assert sorted(chosen.room_label.unique())==result['rooms']
  independent_description(chosen,result['overall'])
  assert set(result['by_room'])==(set(chosen.room_label.unique()) if chosen.room_label.nunique()>1 else set())
  for room,description in result['by_room'].items():independent_description(chosen[chosen.room_label==room],description)
  scene_table=pd.read_csv(dest/'scene_means.csv',float_precision='round_trip',converters={'room_label':str,'direction_label':str})
  assert len(scene_table)==9*chosen.scene.nunique()
  for row in scene_table.itertuples():
   sample=chosen[(chosen.scene==row.scene)&(chosen.selector==row.selector)]
   assert sample.room_label.eq(row.room_label).all() and sample.direction_label.eq(row.direction_label).all()
   for metric in ['lpips','psnr','ssim']:assert abs(sample[metric].mean()-getattr(row,metric))<1e-12
  room_table=pd.read_csv(dest/'room_means.csv',float_precision='round_trip',converters={'room_label':str})
  assert len(room_table)==9*chosen.room_label.nunique()
  for row in room_table.itertuples():
   sample=scene_table[(scene_table.room_label==row.room_label)&(scene_table.selector==row.selector)]
   for metric in ['lpips','psnr','ssim']:assert abs(sample[metric].mean()-getattr(row,metric))<1e-12
  independent_bootstrap=True
 else:
  assert not (dest/'results.json').exists();assert js(dest/'status.json')['inferential_estimate_created'] is False
 result={'passed':True,'independent_original_pipeline':True,'dry_run':a.smoke,'excluded_from_all_analyses':a.smoke,'verified_candidate_rows':count,'verified_selection_rows':verified_choices,'original_top1_pair_tie_diagnostics_verified':diagnostics_verified,'metric_max_abs_diff':maxdiff,'qc':qc_records,'bootstrap_verified':independent_bootstrap,'bootstrap_not_applicable_to_dry_run':a.smoke}
 (dest/'verify.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result),flush=True)
def verify_derived():
 """Reproduce selection and descriptive intervals without private images or model calls."""
 p=argparse.ArgumentParser();p.add_argument('--derived-only',action='store_true');p.add_argument('--run',default='exploratory_01');a=p.parse_args()
 assert Path(a.run).name==a.run
 dest=ROOT/'derived'/a.run
 data=pd.read_csv(dest/'candidates.csv',float_precision='round_trip');chosen=pd.read_csv(dest/'selections.csv',float_precision='round_trip');result=js(dest/'results.json')
 assert not result['hypothesis_test_performed'] and not result['p_values_computed']
 independent_description(chosen,result['overall'])
 count=0
 for scene,x in data.groupby('scene',sort=True):
  for method,ss in chosen[chosen.scene==scene].groupby('selector'):
   previous=2/3
   for row in ss.sort_values('frame').itertuples():
    group=x[x.frame==row.frame];actual=independently_select(group,method,previous,js(ROOT/'PLAN.json')['fixed_gamma']['value'])
    assert actual==row.gamma;previous=actual
    candidate=group[np.isclose(group.gamma,actual,rtol=0,atol=1e-12)].iloc[0]
    for metric in ['lpips','psnr','ssim']:assert candidate[metric]==getattr(row,metric)
    count+=1
 scene_table=pd.read_csv(dest/'scene_means.csv',float_precision='round_trip')
 for row in scene_table.itertuples():
  sample=chosen[(chosen.scene==row.scene)&(chosen.selector==row.selector)]
  for metric in ['lpips','psnr','ssim']:assert abs(sample[metric].mean()-getattr(row,metric))<1e-12
 print(json.dumps({'passed':True,'mode':'derived-only; raw-image metrics not recomputed','candidate_rows':len(data),'selection_rows':count,'scenes':chosen.scene.nunique(),'bootstrap_replicates':result['overall']['bootstrap_replicates'],'bootstrap_verified':True}))
if __name__=='__main__':
 if '--derived-only' in sys.argv:verify_derived()
 else:main()
