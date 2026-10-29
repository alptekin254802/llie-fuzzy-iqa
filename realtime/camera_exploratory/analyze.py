from c_exploratory import *
from descriptive import describe
import argparse,importlib.metadata
import pandas as pd,torch
from lut_candidates import make_tables,lut_candidates
def check_certificate():
 # Portable equivalent of the private certificate gate; synthetic engineering check only.
 from selftest import run
 rng=np.random.default_rng(20261024)
 frames=[np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],256,axis=0).repeat(3,axis=2)]
 frames += [rng.integers(0,256,(256,256,3),dtype=np.uint8) for _ in range(3)]
 assert run(frames,models())['passed']
sys.path.insert(0,str(REPO))
from evaluation.metrics import compute_psnr,compute_ssim
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def metric_set():
 import pyiqa
 assert importlib.metadata.version('pyiqa')==PLAN['scoring']['pyiqa']
 device='cuda' if torch.cuda.is_available() else 'cpu'
 return {m:pyiqa.create_metric(m,device=device) for m in ['lpips','clipiqa+','niqe']},device
def tensor(x,device):return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].to(device)
def choose(rows,method,prev):
 grid=rows[rows.adaptive_candidate].copy()
 if method in ['default','fixed']:
  target=2/3 if method=='default' else PLAN['fixed_gamma']['value'];return rows.loc[np.isclose(rows.gamma,target,rtol=0,atol=1e-12)].iloc[0],1
 vals=-np.abs(grid.brightness.to_numpy()-.45) if method=='brightness' else -grid.lpips.to_numpy() if method=='oracle' else grid[method].to_numpy()*(-1 if method=='niqe' else 1)
 valid=np.isfinite(vals);maximum=vals[valid].max() if valid.any() else np.nan;top=grid[valid&np.isclose(vals,maximum,atol=1e-9,rtol=0)] if valid.any() else grid
 gs=top.gamma.to_numpy();index=np.lexsort((gs,np.round(np.abs(gs-2/3),12),np.round(np.abs(gs-prev),12)))[0]
 return top.iloc[index],len(top)
def score_scenes(scenes,destination,smoke=False):
 mods=models();check_certificate();metrics,device=metric_set();gammas=[*np.array(PLAN['gammas']),2/3];tables=make_tables(gammas);rows=[]
 with torch.inference_mode():
  for folder in scenes:
   meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));assert meta['plan_sha256']==sha(ROOT/'PLAN.json')
   validate_scene_metadata(meta,smoke)
   assert meta['qc']['passed'],'Rejected scene must never enter scoring'
   assert bool(meta['dry_run'])==smoke
   ref=(read_reference(folder/meta['reference_begin']['file'])+read_reference(folder/meta['reference_end']['file']))/2;rt=tensor(ref,device)
   for rec in sorted(meta['frames'],key=lambda r:r['order']):
    assert sha(folder/rec['file'])==rec['sha256'];rgb=resize_rgb8(cv2.imread(str(folder/rec['file'])));inp=rgb.astype(np.float32)/255.0;r=lut_candidates(rgb,tables,mods)
    # Scientific metrics still use original float32 candidates, never RGB8 LUT images.
    r['images']=[gamma_image(inp,g) for g in gammas]
    cache=private_path('candidate_cache',destination.name,meta['scene_id']);cache.mkdir(parents=True,exist_ok=True);cachefile=cache/f"frame_{rec['order']:02d}.npz"
    np.savez_compressed(cachefile,images=np.stack(r['images']),gammas=np.array(gammas),features=r['features'],brightness=r['brightness'],fuzzy=r['fuzzy'],nr_mono=r['nr_mono'])
    with np.load(cachefile) as cached:stored_images=cached['images']
    for j,g in enumerate(gammas):
     im=stored_images[j];it=tensor(im,device);row={'scene':meta['scene_id'],'frame':rec['order'],'stop':rec['stop'],'gamma':float(g),'adaptive_candidate':j<9,'dry_run':smoke,'room_label':meta['room_label'],'direction_label':meta['direction_label'],'input_sha256':rec['sha256'],'candidate_cache_sha256':sha(cachefile),'candidate_index':j,'brightness':float(r['brightness'][j]),**{k:float(r['features'][j,l]) for l,k in enumerate(['entropy','contrast','sharpness'])},**{k:float(r[k][j]) for k in ['fuzzy','nr_mono']}}
     for m,fn in metrics.items():row[m]=float(fn(it,rt).item() if m=='lpips' else fn(it).item())
     row['psnr']=float(compute_psnr(im,ref));row['ssim']=float(compute_ssim(im,ref));assert np.isfinite(row['lpips']);rows.append(row)
    print('scored',meta['scene_id'],rec['order']+1,'/20',flush=True)
 data=pd.DataFrame(rows);data.to_csv(destination/'candidates.csv',index=False)
 selected=[]
 for scene,x in data.groupby('scene',sort=True):
  for method in ['fuzzy','fixed','nr_mono','entropy','brightness','default','oracle','clipiqa+','niqe']:
   prev=2/3
   for frame,y in x.groupby('frame',sort=True):
    chosen,ties=choose(y,method,prev);prev=float(chosen.gamma);selected.append({'scene':scene,'room_label':str(chosen.room_label),'direction_label':str(chosen.direction_label),'frame':int(frame),'selector':method,'gamma':prev,'lpips':float(chosen.lpips),'psnr':float(chosen.psnr),'ssim':float(chosen.ssim),'ties':ties,'dry_run':smoke})
 selected=pd.DataFrame(selected);selected.to_csv(destination/'selections.csv',index=False)
 return selected
def exact_top1_diagnostics(data,dest):
 records=[]
 for (scene,fr),a in data[data.adaptive_candidate].groupby(['scene','frame'],sort=True):
  target=-a.lpips.to_numpy()
  for method in ['fuzzy','nr_mono','entropy','brightness','clipiqa+','niqe','oracle']:
   pred=-np.abs(a.brightness.to_numpy()-.45) if method=='brightness' else target if method=='oracle' else a[method].to_numpy()*(-1 if method=='niqe' else 1)
   if not np.isfinite(pred).all() or len(np.unique(target))<2:
    records.append({'scene':scene,'frame':fr,'selector':method,'top1':None,'pair_acc':None,'pairs':0});continue
   maxima=pred==pred.max();top1=float(np.mean(target[maxima]==target.max()));i,j=np.triu_indices(len(pred),1);valid=target[i]!=target[j];sp=np.sign(pred[i]-pred[j]);sy=np.sign(target[i]-target[j]);pairs=int(valid.sum());correct=float(np.sum((sp==sy)&valid)+.5*np.sum((sp==0)&valid))
   records.append({'scene':scene,'frame':fr,'selector':method,'top1':top1,'pair_acc':correct/max(pairs,1),'pairs':pairs})
 pd.DataFrame(records).to_csv(dest/'exact_top1_diagnostics.csv',index=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--scene',default='dry_run_3');p.add_argument('--run');p.add_argument('--diagnostics-only',action='store_true');a=p.parse_args();validate_exploratory()
 run=a.run or ('smoke_'+a.scene if a.smoke else datetime.datetime.now().strftime('exploratory_%Y%m%d_%H%M%S'))
 if Path(run).name!=run:raise ValueError('Run must be a name, not a path')
 root=private_path('dry_run' if a.smoke else 'accepted');scenes=[]
 for folder in sorted(root.glob('*')):
  if not (folder/'metadata.json').exists():continue
  meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
  if a.smoke and folder.name!=a.scene:continue
  if not meta.get('qc',{}).get('passed'):continue
  if meta.get('amendment_3_sha256')!=sha(ROOT/'PLAN_AMENDMENT_3.json'):continue
  validate_scene_metadata(meta,a.smoke);scenes.append(folder)
 if a.smoke:assert len(scenes)==1,'QC-passing amended dry-run needed for optional scoring smoke test'
 elif len(scenes)<20:raise SystemExit('At least 20 QC-passing exploratory scenes are required; dry runs are excluded. No descriptive result was produced.')
 scope_path=ROOT/'analysis_runs'/run/'INPUTS.json'
 if scope_path.exists():
  scope=json.loads(scope_path.read_text(encoding='utf-8'))
  assert sorted(f.name for f in scenes)==scope['scene_ids'],'Analysis scene set differs from frozen scope'
  assert sorted({json.loads((f/'metadata.json').read_text(encoding='utf-8'))['room_label'] for f in scenes})==scope['rooms']
  assert sha(ROOT/'EXCLUSIONS.json')==scope['exclusions_sha256']
  assert not set(scope['excluded_scene_ids']) & {f.name for f in scenes}
  for f in scenes:assert sha(f/'metadata.json')==scope['scene_metadata_sha256'][f.name]
 dest=DERIVED/run
 if a.diagnostics_only:
  data=pd.read_csv(dest/'candidates.csv',float_precision='round_trip',converters={'room_label':str,'direction_label':str});assert data.dry_run.eq(a.smoke).all();exact_top1_diagnostics(data,dest);return
 dest.mkdir(parents=True,exist_ok=False)
 qc_records=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((DERIVED/'qc').glob('*.json'))]
 save(dest/'qc_records.json',qc_records)
 selected=score_scenes(scenes,dest,a.smoke)
 exact_top1_diagnostics(pd.read_csv(dest/'candidates.csv',float_precision='round_trip',converters={'room_label':str,'direction_label':str}),dest)
 if a.smoke:save(dest/'status.json',{'dry_run':True,'excluded_from_all_analyses':True,'inferential_estimate_created':False,'descriptive_result_created':False,'candidate_rows':200*len(scenes),'selected_rows':len(selected),'metrics':['lpips','psnr','ssim','clipiqa+','niqe']})
 else:
  result=describe(selected,dest);result['amendment_3_sha256']=sha(ROOT/'PLAN_AMENDMENT_3.json');save(dest/'results.json',result)
 print('Derived tables:',dest,flush=True)
if __name__=='__main__':main()
