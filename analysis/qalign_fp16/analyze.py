"""Reproduce fp16 accuracy, precision checks and paired NR-free contrasts from archived scores."""
from pathlib import Path
import os
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,required=True,help='Prepared image manifest with human scores and frozen nr_free predictions')
parser.add_argument('--fp16-scores',type=Path,required=True)
parser.add_argument('--fourbit-scores',type=Path,required=True)
parser.add_argument('--fourbit-results',type=Path,required=True,help='Archived 4-bit report, retained only as timing source and accuracy cross-check')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,hashlib,re,csv,datetime
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
B=2000;SEED=20261018

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p):return pd.read_csv(p,dtype={'scene':str},float_precision='round_trip')
def rho(a,b):return 0. if len(np.unique(a))<2 or len(np.unique(b))<2 else float(spearmanr(a,b).statistic)
def interval(v,p):return {'estimate':float(p),'ci95':[float(x) for x in np.quantile(v,[.025,.975])]}
def stats(x):return dict(mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.quantile(x,.95)))

def main():
 manifest=read(args.manifest)
 scores=read(args.fp16_scores);four=read(args.fourbit_scores);prior=json.loads(args.fourbit_results.read_text('utf-8'))
 decision={'selected_datasets':[ds for ds in ['squarelol','rlie','lieq'] if ds in set(scores.dataset)],'omitted_datasets':sorted(set(manifest.dataset)-set(scores.dataset))}
 assert not scores.duplicated(['dataset','name']).any()
 assert np.isfinite(scores.score).all()
 data=manifest.merge(scores,on=['dataset','name','scene'],validate='one_to_one',how='inner')
 data=data.merge(four[['dataset','name','scene','score','rgb256_sha256']].rename(columns={'score':'score_4bit','rgb256_sha256':'rgb256_sha256_4bit'}),on=['dataset','name','scene'],validate='one_to_one')
 assert (data.rgb256_sha256==data.rgb256_sha256_4bit).all()
 data.to_csv(O/'joined.csv',index=False)
 report={'scores_sha256':sha(args.fp16_scores),'manifest_sha256':sha(args.manifest),'fourbit_scores_sha256':sha(args.fourbit_scores),'analysis_sha256':sha(__file__),'bootstrap':B,'bootstrap_seed':SEED,
  'decision_metric':'mean within-scene human SRCC','threshold':.02,'precision':'fp16, no4/8-bit weights, GPU+CPU dispatch',
  'timing_notice':'Offloaded fp16 timing is feasibility-only; paper4bit GPU1.18s/image remains unchanged.',
  'selected_datasets':decision['selected_datasets'],'omitted_datasets':decision['omitted_datasets'],'datasets':{}}
 scene_rows=[];summary=[];diffrows=[]
 for ds in decision['selected_datasets']:
  x=data[data.dataset==ds].sort_values('name').reset_index(drop=True)
  assert len(x)==len(manifest[manifest.dataset==ds]),('incomplete dataset',ds)
  scene_ids=sorted(x.scene.unique());indices=[np.flatnonzero(x.scene.to_numpy()==s) for s in scene_ids]
  a=x.score.to_numpy();b=x.score_4bit.to_numpy();c=x.nr_free.to_numpy();human=x.mos.to_numpy()
  key=np.random.default_rng(20261021).random(len(x))
  arrays={v:{k:[] for k in ['within_scene_srcc','pair_numerator','pair_denominator','top1','top1_uniform_expectation']} for v in ['fp16','fourbit','nr_free']}
  skipped=[];valid_ids=[]
  for scene,ix in zip(scene_ids,indices):
   y=human[ix]
   if np.ptp(y)==0:skipped.append(scene);continue
   valid_ids.append(ix);i,j=np.triu_indices(len(ix),1);keep=y[i]!=y[j];i=i[keep];j=j[keep];sign=np.sign(y[i]-y[j]);den=len(i)
   for version,scores_all in [('fp16',a),('fourbit',b),('nr_free',c)]:
    s=scores_all[ix];diff=np.sign(s[i]-s[j]);num=float((diff==sign).sum()+.5*(diff==0).sum())
    maxima=np.flatnonzero(s==s.max());pick=maxima[np.argmax(key[ix][maxima])]
    values=dict(within_scene_srcc=rho(s,y),pair_numerator=num,pair_denominator=den,top1=float(y[pick]==y.max()),top1_uniform_expectation=float((y[maxima]==y.max()).mean()))
    for k,v in values.items():arrays[version][k].append(v)
    scene_rows.append(dict(dataset=ds,scene=scene,version=version,n_images=len(ix),predicted_top_ties=len(maxima),prediction_tied_pairs=int((diff==0).sum()),constant_prediction_scene=int(np.ptp(s)==0),**values))
  n=len(valid_ids);assert n>0
  for version in arrays:
   for metric in arrays[version]:arrays[version][metric]=np.asarray(arrays[version][metric])
  metrics=['within_scene_srcc','pair_accuracy','top1','top1_uniform_expectation','pooled_srcc']
  boot={v:{m:[] for m in metrics} for v in arrays};between=[];rng=np.random.default_rng(SEED)
  for _ in range(B):
   draw=rng.integers(n,size=n);ii=np.concatenate([valid_ids[k] for k in draw])
   between.append(rho(a[ii],b[ii]))
   for version,scores_all in [('fp16',a),('fourbit',b),('nr_free',c)]:
    z=arrays[version]
    for m in ['within_scene_srcc','top1','top1_uniform_expectation']:boot[version][m].append(float(z[m][draw].mean()))
    boot[version]['pair_accuracy'].append(float(z['pair_numerator'][draw].sum()/z['pair_denominator'][draw].sum()))
    boot[version]['pooled_srcc'].append(rho(scores_all[ii],human[ii]))
  absolute={};paired={}
  for version,scores_all in [('fp16',a),('fourbit',b),('nr_free',c)]:
   z=arrays[version];absolute[version]={}
   points={m:z[m].mean() for m in ['within_scene_srcc','top1','top1_uniform_expectation']}
   points['pair_accuracy']=z['pair_numerator'].sum()/z['pair_denominator'].sum();points['pooled_srcc']=rho(scores_all,human)
   for m,p in points.items():
    absolute[version][m]=interval(boot[version][m],p)
    summary.append(dict(dataset=ds,version=version,metric=m,estimate=float(p),ci_low=absolute[version][m]['ci95'][0],ci_high=absolute[version][m]['ci95'][1]))
  for m in metrics:
   dd=np.array(boot['fp16'][m])-np.array(boot['fourbit'][m]);point=absolute['fp16'][m]['estimate']-absolute['fourbit'][m]['estimate']
   paired[m]=interval(dd,point);diffrows.append(dict(dataset=ds,metric=m,difference=point,ci_low=paired[m]['ci95'][0],ci_high=paired[m]['ci95'][1]))
  paired_nr={}
  for m in metrics:
   dd=np.array(boot['fp16'][m])-np.array(boot['nr_free'][m]);point=absolute['fp16'][m]['estimate']-absolute['nr_free'][m]['estimate']
   paired_nr[m]=interval(dd,point)
   paired_nr[m].update(p_two_sided_null_centered=float((1+np.count_nonzero(np.abs(dd-point)>=abs(point)))/(B+1)),replicates=B,seed=SEED)
  crosscheck={}
  for m in ['within_scene_srcc','pooled_srcc','pair_accuracy']:
   old=prior['datasets'][ds]['metrics']['qalign'][m];new=absolute['fourbit'][m]['estimate'];crosscheck[m]=new-old
   assert abs(new-old)<1e-10,('archived 4-bit mismatch',ds,m,new,old)
  report['datasets'][ds]={'n_images':len(x),'n_scenes':n,'constant_human_scenes_excluded':skipped,'metrics':absolute,'fp16_minus_4bit':paired,'fp16_minus_nr_free':paired_nr,
    'image_level_fp16_4bit_srcc':interval(between,rho(a,b)),'image_level_fp16_4bit_pearson':float(pearsonr(a,b).statistic),
    'threshold_exceeded':bool(abs(paired['within_scene_srcc']['estimate'])>.02),'fourbit_recheck_difference':crosscheck,
    'fp16_offload_wall_seconds_NOT_cost_benchmark':stats(x.wall_ms.to_numpy()/1000)}
  print(ds,json.dumps(report['datasets'][ds]),flush=True)
 pd.DataFrame(scene_rows).to_csv(O/'per_scene.csv',index=False);pd.DataFrame(summary).to_csv(O/'metrics.csv',index=False);pd.DataFrame(diffrows).to_csv(O/'paired_differences.csv',index=False)
 report['any_threshold_exceeded']=any(v['threshold_exceeded'] for v in report['datasets'].values())

 save(O/'precision_results.json',report)
 # Accuracy and timing are explicitly distinct sources. Never publish offload latency as the cost benchmark.
 publication={'metric':'Q-Align / OneAlign, fp16 accuracy; 4-bit GPU timing','orientation':'higher is better',
  'comparison_direction':'Q-Align minus NR-free','replicates':B,'seed':SEED,
  'raw_scores_sha256':sha(args.fp16_scores),'fourbit_timing_source_sha256':sha(args.fourbit_results),
  'accuracy_source':'qalign_fp16_scores.csv','timing_source':'qalign_results.json (archived 4-bit run)',
  'timing_all_images_ms':prior['timing_all_images_ms'],'datasets':{}}
 for ds,r in report['datasets'].items():
  absolute={}
  for name,version in [('qalign','fp16'),('nr_free','nr_free')]:
   absolute[name]={}
   for metric,values in r['metrics'][version].items():
    absolute[name][metric]=values['estimate'];absolute[name][metric+'_ci95']=values['ci95']
   scenes=[z for z in scene_rows if z['dataset']==ds and z['version']==version]
   absolute[name]['prediction_tied_pairs']=sum(z['prediction_tied_pairs'] for z in scenes)
   absolute[name]['constant_prediction_scenes']=sum(z['constant_prediction_scene'] for z in scenes)
  publication['datasets'][ds]={'n_images':r['n_images'],'n_scenes':r['n_scenes'],'metrics':absolute,
   'qalign_minus_nr_free':r['fp16_minus_nr_free'],'inference_ms':prior['datasets'][ds]['inference_ms'],
   'inference_precision':'4-bit NF4, GPU; not the fp16 offload run',
   'image_level_fp16_4bit_srcc':r['image_level_fp16_4bit_srcc']}
 save(O/'publication_results.json',publication)
 print('Wrote precision_results.json and publication_results.json; offload timing is not the benchmark.')
if __name__=='__main__':main()
