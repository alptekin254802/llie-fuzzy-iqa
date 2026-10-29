"""Complete RichIQA paired scene analysis; no fitting or manual result entry."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from scipy.stats import spearmanr
from analysis.studies.config import OUT
O=OUT;B=2000;SEED=20261019;DS=['lol','squarelol','lieq','rlie']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rho(a,b):return float(spearmanr(a,b).statistic) if len(np.unique(a))>1 and len(np.unique(b))>1 else 0.
def summary(v):return {'n':len(v),'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'mean':float(np.mean(v))}
def contrast(draw,estimate):return {'estimate':float(estimate),'ci95':np.percentile(draw,[2.5,97.5]).tolist(),'p_two_sided_null_centered':float((1+np.count_nonzero(np.abs(np.asarray(draw)-estimate)>=abs(estimate)))/(B+1)),'replicates':B,'seed':SEED}
def main():
 state=json.loads((O/'richiqa_run.json').read_text());assert state['status']=='completed'
 m=pd.read_csv(O/'richiqa_manifest.csv',dtype={'scene':str},float_precision='round_trip');s=pd.read_csv(O/'richiqa_scores.csv',dtype={'scene':str},float_precision='round_trip');assert len(s)==len(m)==7865 and not s.duplicated(['dataset','name']).any()
 d=m.merge(s,on=['dataset','name','scene','rgb256_sha256'],validate='one_to_one');assert len(d)==7865 and np.isfinite(d.score).all()
 probs=d[[f'p{i}' for i in range(1,6)]].to_numpy();assert np.allclose(probs.sum(1),1,rtol=0,atol=1e-5);assert np.allclose(probs@np.arange(1,6),d.score,rtol=0,atol=1e-12)
 res={'plan_sha256':sha(O/'richiqa_PLAN.md'),'manifest_sha256':sha(O/'richiqa_manifest.csv'),'raw_sha256':sha(O/'richiqa_scores.csv'),'analysis_sha256':sha(Path(__file__)),'orientation':'higher is better','post_hoc_diagnostics':{'lol_pair_accuracy':'Computed as an auxiliary diagnostic; not a prespecified LOL endpoint and not added to manuscript tables'},'difference_direction':'RichIQA minus NR-free','bootstrap':{'replicates':B,'seed':SEED,'unit':'scene','ci':'percentile 95%','p':'two-sided null-centered with plus-one correction; unadjusted','lol_fold_mean':'resample scenes within each fixed fold, preserve fold sizes'},'datasets':{}}
 per=[]
 for ds in DS:
  x=d[d.dataset==ds].reset_index(drop=True);ids=[g.index.to_numpy() for _,g in x.groupby('scene',sort=True)];names=sorted(x.scene.unique());y=x.target.to_numpy();pred={'richiqa':x.score.to_numpy(),'nr_free':x.nr_free.to_numpy()};scene_r={k:[] for k in pred};correct={k:[] for k in pred};pairs=[];valid=[];ties={k:0 for k in pred};constant={k:0 for k in pred};target_ties=0
  for name,ii in zip(names,ids):
   yy=y[ii];valid.append(len(np.unique(yy))>1);a,b=np.triu_indices(len(ii),1);mask=yy[a]!=yy[b];target_ties+=int((~mask).sum());a=a[mask];b=b[mask];sy=np.sign(yy[a]-yy[b]);pairs.append(len(a));row={'dataset':ds,'scene':name,'valid_target':valid[-1],'pairs':len(a),'fold':int(x.loc[ii[0],'fold'])}
   for k,p in pred.items():
    rr=rho(p[ii],yy) if valid[-1] else np.nan;ss=np.sign(p[ii][a]-p[ii][b]);nn=float(np.sum(ss==sy)+.5*np.sum(ss==0));scene_r[k].append(rr);correct[k].append(nn);ties[k]+=int(np.sum(ss==0));constant[k]+=int(len(np.unique(p[ii]))<2);row[k+'_srcc']=rr;row[k+'_pair_correct']=nn
   per.append(row)
  for k in pred:scene_r[k]=np.asarray(scene_r[k]);correct[k]=np.asarray(correct[k])
  pairs=np.asarray(pairs);metrics={k:{'pooled_srcc':rho(p,y),'within_scene_srcc':float(np.nanmean(scene_r[k])),'pair_accuracy':float(correct[k].sum()/pairs.sum()),'prediction_tied_pairs':ties[k],'constant_prediction_scenes':constant[k]} for k,p in pred.items()}
  measure=['pooled_srcc','within_scene_srcc','pair_accuracy'];foldids={}
  if ds=='lol':
   foldids={f:np.array([j for j,ii in enumerate(ids) if int(x.loc[ii[0],'fold'])==f]) for f in sorted(x.fold.unique())}
   assert len(foldids)==5 and all(len(z)==97 for z in foldids.values());assert all(x.loc[ii,'fold'].nunique()==1 for ii in ids)
   for k,p in pred.items():
    rr=[rho(p[x.fold.to_numpy()==f],y[x.fold.to_numpy()==f]) for f in foldids];metrics[k].update({'fold_srcc':rr,'fold_mean_srcc':float(np.mean(rr)),'fold_std_srcc':float(np.std(rr,ddof=0))})
   measure+=['fold_mean_srcc']
  boot={k:{z:[] for z in measure} for k in pred};rng=np.random.default_rng(SEED)
  for _ in range(B):
   sel=rng.integers(len(ids),size=len(ids));jj=np.concatenate([ids[z] for z in sel])
   if ds=='lol':fj=[np.concatenate([ids[z] for z in rng.choice(v,size=len(v),replace=True)]) for v in foldids.values()]
   for k,p in pred.items():
    boot[k]['pooled_srcc'].append(rho(p[jj],y[jj]));boot[k]['within_scene_srcc'].append(float(np.nanmean(scene_r[k][sel])));boot[k]['pair_accuracy'].append(float(correct[k][sel].sum()/pairs[sel].sum()))
    if ds=='lol':boot[k]['fold_mean_srcc'].append(float(np.mean([rho(p[z],y[z]) for z in fj])))
  delta={}
  for z in measure:
   for k in pred:metrics[k][z+'_ci95']=np.percentile(boot[k][z],[2.5,97.5]).tolist()
   delta[z]=contrast(np.asarray(boot['richiqa'][z])-np.asarray(boot['nr_free'][z]),metrics['richiqa'][z]-metrics['nr_free'][z])
  res['datasets'][ds]={'n_images':len(x),'n_scenes':len(ids),'excluded_constant_target_scenes':int(np.sum(~np.asarray(valid))),'target_tied_pairs_skipped':target_ties,'valid_pairs':int(pairs.sum()),'metrics':metrics,'richiqa_minus_nr_free':delta,'gpu_ms':summary(x.ms)};print(ds,json.dumps(metrics),flush=True)
 cpu=pd.read_csv(O/'richiqa_cpu.csv',float_precision='round_trip');assert len(cpu)==300 and cpu.groupby('session').size().eq(100).all()
 cpu=cpu.merge(s[['dataset','name','score']],on=['dataset','name'],suffixes=('_cpu','_gpu'),validate='many_to_one');error=np.abs(cpu.score_cpu-cpu.score_gpu);assert np.allclose(cpu.score_cpu,cpu.score_gpu,rtol=0,atol=1e-4)
 res['timing']={'cpu_ms':summary(cpu.ms),'gpu_ms':summary(s.ms),'cpu_sessions':{str(k):summary(z.ms) for k,z in cpu.groupby('session')},'cpu_gpu_agreement':{'n_calls':len(cpu),'n_images':len(cpu[['dataset','name']].drop_duplicates()),'max_abs_difference':float(error.max()),'tolerance':1e-4},'environment':json.loads((O/'richiqa_environment.json').read_text())}
 res['sources']=json.loads((O/'richiqa_sources.json').read_text());pd.DataFrame(per).to_csv(O/'richiqa_per_scene.csv',index=False);d.to_csv(O/'richiqa_joined.csv',index=False);(O/'richiqa_results.json').write_text(json.dumps(res,indent=2,allow_nan=False),encoding='utf-8');print('All results saved',flush=True)
if __name__=='__main__':main()
