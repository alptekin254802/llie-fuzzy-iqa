"""Scene-cluster paired analysis; all estimates written from scores to JSON."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,warnings
from scipy.stats import spearmanr
from analysis.studies.config import OUT
O=OUT;B=2000;SEED=20261018
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rho(a,b):
 if len(np.unique(a))<2 or len(np.unique(b))<2:return 0.
 return float(spearmanr(a,b).statistic)
def interval(draw,estimate):
 draw=np.asarray(draw);lo,hi=np.percentile(draw,[2.5,97.5]);return {'estimate':float(estimate),'ci95':[float(lo),float(hi)]}
def difference(draw,estimate):
 d=interval(draw,estimate);d.update({'p_two_sided_null_centered':float((1+np.count_nonzero(np.abs(np.asarray(draw)-estimate)>=abs(estimate)))/(B+1)),'replicates':B,'seed':SEED});return d
def stats(v):return {'n':len(v),'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'mean':float(np.mean(v))}
def main():
 scores=pd.read_csv(O/'qalign_scores.csv',dtype={'scene':str});manifest=pd.read_csv(O/'qalign_manifest.csv',dtype={'scene':str})
 assert not scores.duplicated(['dataset','name']).any();assert len(scores)==len(manifest)==5440
 data=manifest.merge(scores,on=['dataset','name','scene'],validate='one_to_one');assert len(data)==5440 and np.isfinite(data.score).all()
 report={'plan_sha256':sha(O/'qalign_PLAN.md'),'raw_scores_sha256':sha(O/'qalign_scores.csv'),'manifest_sha256':sha(O/'qalign_manifest.csv'),'script_sha256':sha(Path(__file__)),'metric':'Q-Align / OneAlign, 4-bit NF4','orientation':'higher is better','comparison_direction':'Q-Align minus NR-free','uncertainty':'paired whole-scene bootstrap, percentile 95% CI; null-centered two-sided p-values, unadjusted','replicates':B,'seed':SEED,'quantization_caveat':'4-bit quantization can change scores, rankings and ties relative to full precision; no full-precision equivalence is claimed.','datasets':{}}
 per_scene=[]
 for ds,x in data.groupby('dataset',sort=False):
  x=x.reset_index(drop=True);g=x.scene.to_numpy();y=x.mos.to_numpy();a=x.score.to_numpy();b=x.nr_free.to_numpy();scenes=sorted(x.scene.unique());ids=[np.flatnonzero(g==s) for s in scenes]
  valid=[];ar=[];br=[];an=[];bn=[];pn=[];ties_a=0;ties_b=0;const_a=0;const_b=0;target_ties=0
  for s,ii in zip(scenes,ids):
   yy=y[ii];aa=a[ii];bb=b[ii]
   if len(np.unique(yy))<2:continue
   i,j=np.triu_indices(len(ii),1);different=yy[i]!=yy[j];target_ties+=int((~different).sum());i=i[different];j=j[different];sy=np.sign(yy[i]-yy[j]);sa=np.sign(aa[i]-aa[j]);sb=np.sign(bb[i]-bb[j]);den=len(i)
   na=float(np.sum(sa==sy)+.5*np.sum(sa==0));nb=float(np.sum(sb==sy)+.5*np.sum(sb==0))
   ra=rho(aa,yy);rb=rho(bb,yy);ar.append(ra);br.append(rb);an.append(na);bn.append(nb);pn.append(den);valid.append(ii)
   ta=int(np.sum(sa==0));tb=int(np.sum(sb==0));ties_a+=ta;ties_b+=tb;const_a+=int(len(np.unique(aa))==1);const_b+=int(len(np.unique(bb))==1)
   per_scene.append({'dataset':ds,'scene':s,'qalign_srcc':ra,'nr_free_srcc':rb,'pair_denominator':den,'qalign_pair_numerator':na,'nr_free_pair_numerator':nb,'qalign_tied_pairs':ta,'nr_free_tied_pairs':tb})
  ar=np.asarray(ar);br=np.asarray(br);an=np.asarray(an);bn=np.asarray(bn);pn=np.asarray(pn)
  assert len(valid)==len(ids),'Prespecified datasets expected variable MOS in every scene; report explicitly if not'
  absolute={'qalign':{'within_scene_srcc':float(ar.mean()),'pooled_srcc':rho(a,y),'pair_accuracy':float(an.sum()/pn.sum()),'constant_prediction_scenes':const_a,'prediction_tied_pairs':ties_a},'nr_free':{'within_scene_srcc':float(br.mean()),'pooled_srcc':rho(b,y),'pair_accuracy':float(bn.sum()/pn.sum()),'constant_prediction_scenes':const_b,'prediction_tied_pairs':ties_b}}
  boot={model:{m:[] for m in ['within_scene_srcc','pooled_srcc','pair_accuracy']} for model in ['qalign','nr_free']};rng=np.random.default_rng(SEED)
  for _ in range(B):
   selection=rng.integers(len(ids),size=len(ids));jj=np.concatenate([ids[k] for k in selection]);den=pn[selection].sum()
   for model,rr,nn,pp in [('qalign',ar,an,a),('nr_free',br,bn,b)]:
    boot[model]['within_scene_srcc'].append(float(rr[selection].mean()));boot[model]['pair_accuracy'].append(float(nn[selection].sum()/den));boot[model]['pooled_srcc'].append(rho(pp[jj],y[jj]))
  diff={}
  for m in boot['qalign']:
   for model in boot:absolute[model][m+'_ci95']=interval(boot[model][m],absolute[model][m])['ci95']
   dd=np.asarray(boot['qalign'][m])-np.asarray(boot['nr_free'][m]);diff[m]=difference(dd,absolute['qalign'][m]-absolute['nr_free'][m])
  report['datasets'][ds]={'n_images':len(x),'n_scenes':len(ids),'excluded_constant_target_scenes':len(ids)-len(valid),'valid_pairs':int(pn.sum()),'target_tied_pairs_skipped':target_ties,'metrics':absolute,'qalign_minus_nr_free':diff,'inference_ms':stats(x.inference_ms),'prepare_ms':stats(x.prepare_ms),'h2d_ms':stats(x.h2d_ms)}
  print(ds,json.dumps(absolute),flush=True)
 pd.DataFrame(per_scene).to_csv(O/'qalign_per_scene.csv',index=False)
 data.to_csv(O/'qalign_joined.csv',index=False)
 report['timing_all_images_ms']=stats(data.inference_ms);report['environment']=json.loads((O/'qalign_environment.json').read_text())
 report['windows_gpu_memory_sample']=json.loads((O/'qalign_gpu_memory_sample.json').read_text(encoding='utf-8-sig')) if (O/'qalign_gpu_memory_sample.json').exists() else None
 (O/'qalign_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
