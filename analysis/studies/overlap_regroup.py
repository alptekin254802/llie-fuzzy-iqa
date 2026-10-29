"""Recompute overlap subgroup summaries from fixed stored scores after classification commit."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\r\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--classification-commit',required=True);a=ap.parse_args()
 rec=ROOT/'results/studies/records';path=rec/'squarelol_lol_scene_matches.csv'
 saved=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'show',a.classification_commit+':'+path.relative_to(ROOT).as_posix()]).decode().replace('\r\n','\n')
 assert path.read_text(encoding='utf-8')==saved
 ov=pd.read_csv(path);assert len(ov)==290 and ov.shared_source.notna().all() and ov.status.eq('verified_normal_light_thumbnail').all()
 p=pd.read_csv(rec/'squarelol_frozen_de_predictions.csv').rename(columns={'prediction':'nr_free'})
 p=p.merge(pd.read_csv(rec/'squarelol_frozen_linear_predictions.csv')[['name','prediction']].rename(columns={'prediction':'linear'}),on='name',validate='one_to_one')
 p=p.merge(pd.read_csv(rec/'squarelol_existing_metric_predictions.csv'),on=['name','scene'],validate='one_to_one')
 lmm=pd.read_csv(ROOT/'results/lmm_extra/joined.csv');lmm=lmm[lmm.dataset.eq('squarelol')]
 p=p.merge(lmm[['name','scene','mos','qalign','vqa2','vital']],on=['name','scene'],validate='one_to_one',suffixes=('','_lmm'))
 assert len(p)==2900 and set(p.scene)==set(ov.squarelol)
 assert np.allclose(p.mos,p.mos_lmm,rtol=0,atol=1e-12) and np.array_equal(p.qalign_fp16,p.qalign)
 models=['nr_free','linear','lpips_alex','qalign_fp16','vqa2','vital'];per=[]
 for scene,v in p.groupby('scene'):
  for m in models:per.append({'scene':scene,'model':m,'within_scene_srcc':float(spearmanr(v[m],v.mos).statistic)})
 per=pd.DataFrame(per);assert np.isfinite(per.within_scene_srcc).all()
 result={'method':'Normal-light 48x32 equalized grayscale thumbnail, maximum correlation >= 0.9','classification_commit':a.classification_commit,'threshold':.9,'counts':{},'groups':{},'sources':{f.relative_to(ROOT).as_posix():sha(f) for f in [path,rec/'squarelol_frozen_de_predictions.csv',rec/'squarelol_frozen_linear_predictions.csv',rec/'squarelol_existing_metric_predictions.csv',ROOT/'results/lmm_extra/joined.csv']}}
 extra={}
 for group,flag in [('shared',True),('nonoverlap',False)]:
  scenes=set(ov.loc[ov.shared_source.eq(flag),'squarelol']);d=p[p.scene.isin(scenes)];result['counts'][group]=len(scenes);result['groups'][group]={}
  for m in models:result['groups'][group][m]={'pooled_srcc':float(spearmanr(d[m],d.mos).statistic),'within_scene_srcc':float(per[per.scene.isin(scenes)&per.model.eq(m)].within_scene_srcc.mean()),'n_scenes':len(scenes)}
  extra[group]={m:{k:v for k,v in result['groups'][group][m].items() if k!='n_scenes'} for m in ['vqa2','vital']}
 dump(ROOT/'results/studies/overlap_transfer_verified.json',result);dump(ROOT/'results/lmm_extra/overlap_verified.json',extra)
 per.merge(ov[['squarelol','shared_source']].rename(columns={'squarelol':'scene'}),on='scene',validate='many_to_one').to_csv(ROOT/'results/studies/records/overlap_scene_srcc.csv',index=False,lineterminator='\r\n')
 print(json.dumps({'counts':result['counts'],'groups':result['groups']},indent=2))
if __name__=='__main__':main()
