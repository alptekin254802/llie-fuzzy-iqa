"""Reproduce released LMM comparisons after loading local human labels; no model inference."""
from pathlib import Path
import argparse, json, os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
 os.environ[key]='1'
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from analysis.human_labels import attach_local_mos, read_csv, local_output
ROOT=Path(__file__).resolve().parents[2]
B=2000;SEED=20261018

def rho(a,b):return 0. if len(np.unique(a))<2 or len(np.unique(b))<2 else float(spearmanr(a,b).statistic)
def interval(v,p):return {'estimate':float(p),'ci95':[float(x) for x in np.quantile(v,[.025,.975])]}

def analyse(data, output):
 methods={'nr_free':'nr_free','qalign':'qalign','vqa2':'vqa2','vital':'vital'}
 report={'bootstrap':B,'bootstrap_seed':SEED,'top1_seed':20261021,'datasets':{}}
 summary=[];diffrows=[];scene_rows=[]
 for ds in ['squarelol','rlie','lieq']:
  x=data[data.dataset==ds].sort_values('name').reset_index(drop=True)
  human=x.mos.to_numpy();key=np.random.default_rng(20261021).random(len(x))
  arrays={v:{k:[] for k in ['within_scene_srcc','pair_numerator','pair_denominator','top1','top1_uniform_expectation']} for v in methods}
  valid_ids=[];skipped=[]
  for scene in sorted(x.scene.unique()):
   ix=np.flatnonzero(x.scene.to_numpy()==scene);y=human[ix]
   if np.ptp(y)==0:skipped.append(scene);continue
   valid_ids.append(ix);i,j=np.triu_indices(len(ix),1);keep=y[i]!=y[j];i=i[keep];j=j[keep];sign=np.sign(y[i]-y[j]);den=len(i)
   for version,col in methods.items():
    s=x[col].to_numpy()[ix];diff=np.sign(s[i]-s[j]);num=float((diff==sign).sum()+.5*(diff==0).sum())
    maxima=np.flatnonzero(s==s.max());pick=maxima[np.argmax(key[ix][maxima])]
    values=dict(within_scene_srcc=rho(s,y),pair_numerator=num,pair_denominator=den,top1=float(y[pick]==y.max()),top1_uniform_expectation=float((y[maxima]==y.max()).mean()))
    for k,v in values.items():arrays[version][k].append(v)
    scene_rows.append(dict(dataset=ds,scene=scene,version=version,n_images=len(ix),predicted_top_ties=len(maxima),**values))
  n=len(valid_ids)
  for version in arrays:
   for metric in arrays[version]:arrays[version][metric]=np.asarray(arrays[version][metric])
  metrics=['within_scene_srcc','pair_accuracy','top1','top1_uniform_expectation','pooled_srcc']
  boot={v:{m:[] for m in metrics} for v in methods};rng=np.random.default_rng(SEED)
  for _ in range(B):
   draw=rng.integers(n,size=n);ii=np.concatenate([valid_ids[k] for k in draw])
   for version,col in methods.items():
    z=arrays[version]
    for m in ['within_scene_srcc','top1','top1_uniform_expectation']:boot[version][m].append(float(z[m][draw].mean()))
    boot[version]['pair_accuracy'].append(float(z['pair_numerator'][draw].sum()/z['pair_denominator'][draw].sum()))
    boot[version]['pooled_srcc'].append(rho(x[col].to_numpy()[ii],human[ii]))
  absolute={}
  for version,col in methods.items():
   z=arrays[version];absolute[version]={}
   points={m:z[m].mean() for m in ['within_scene_srcc','top1','top1_uniform_expectation']}
   points['pair_accuracy']=z['pair_numerator'].sum()/z['pair_denominator'].sum();points['pooled_srcc']=rho(x[col].to_numpy(),human)
   for m,p in points.items():
    absolute[version][m]=interval(boot[version][m],p)
    summary.append(dict(dataset=ds,version=version,metric=m,estimate=float(p),ci_low=absolute[version][m]['ci95'][0],ci_high=absolute[version][m]['ci95'][1]))
  paired={}
  for model in methods:
   if model in ['nr_free','qalign']:continue
   for baseline in ['nr_free','qalign']:
    pair=model+'_minus_'+baseline;paired[pair]={}
    for m in metrics:
     dd=np.array(boot[model][m])-np.array(boot[baseline][m]);point=absolute[model][m]['estimate']-absolute[baseline][m]['estimate']
     paired[pair][m]=interval(dd,point);diffrows.append(dict(dataset=ds,comparison=pair,metric=m,difference=point,ci_low=paired[pair][m]['ci95'][0],ci_high=paired[pair][m]['ci95'][1]))
  report['datasets'][ds]={'n_images':len(x),'n_scenes':n,'constant_human_scenes_excluded':skipped,'metrics':absolute,'paired_differences':paired}
 pd.DataFrame(summary).to_csv(output/'metrics.csv',index=False)
 pd.DataFrame(diffrows,columns=['dataset','comparison','metric','difference','ci_low','ci_high']).to_csv(output/'paired_differences.csv',index=False)
 pd.DataFrame(scene_rows).to_csv(output/'per_scene.csv',index=False)
 return report

def check_values(actual, expected, path=''):
 if isinstance(expected,dict):
  for key,value in expected.items():check_values(actual[key],value,path+'/'+key)
 elif isinstance(expected,list):
  assert len(actual)==len(expected),path
  for i,(a,b) in enumerate(zip(actual,expected)):check_values(a,b,path+'/'+str(i))
 elif isinstance(expected,(int,float)):
  assert np.isclose(actual,expected,rtol=0,atol=1e-12),(path,actual,expected)
 else:assert actual==expected,(path,actual,expected)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,default=ROOT/'results/lmm_extra/joined.csv')
 p.add_argument('--lieq-root',type=Path);p.add_argument('--rlie-root',type=Path)
 p.add_argument('--output',type=Path,default=ROOT/'study_work/lmm_extra')
 a=p.parse_args();output=local_output(a.output/'results.json').parent
 data=attach_local_mos(read_csv(a.input),a.lieq_root,a.rlie_root)
 report=analyse(data,output)
 expected=json.loads((ROOT/'results/lmm_extra/results.json').read_text(encoding='utf8'))
 check_values(report['datasets'],expected['datasets'])
 for filename in ['per_scene.csv','paired_differences.csv']:
  pd.testing.assert_frame_equal(read_csv(output/filename),read_csv(ROOT/'results/lmm_extra'/filename),check_exact=False,rtol=0,atol=1e-12)
 (output/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print('PASS: all released point estimates, scene-bootstrap intervals, paired differences and per-scene records reproduced.')
if __name__=='__main__':main()
