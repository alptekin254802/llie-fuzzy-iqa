"""Select qualitative scenes from stored scores before opening images."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
R=args.output.resolve();R.mkdir(parents=True,exist_ok=True)
P=Path(__file__).resolve().parents[2]/'results'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=P/'figure_protocols/rule_selection.json';plan=json.loads(plan_path.read_text())
for n,h in plan['inputs_sha256'].items():assert sha(P/n)==h,n
oof=pd.read_csv(P/'nr_fuzzy_oof.csv')
metrics=pd.read_csv(P/'perceptual_metrics.csv')[['filename','method','lpips']]
psnr=pd.read_csv(P/'enhancement_metrics.csv')[['filename','method','psnr']]
a=oof.merge(metrics,on=['filename','method'],validate='one_to_one').merge(psnr,on=['filename','method'],validate='one_to_one')
assert len(a)==2425 and a.filename.nunique()==485
assert a.groupby('filename').size().eq(5).all() and a.groupby('filename').fold.nunique().eq(1).all()
v=pd.read_csv(P/'validation_predictions.csv')
t=a.merge(v[['filename','method','oof_nr_free']],on=['filename','method'],validate='one_to_one')
assert np.allclose(t.prediction,t.oof_nr_free,atol=1e-12,rtol=0)
def scene_num(n):return int(re.search(r'\d+',n)[0])
def correlations(df,scene,x,y):
 rows=[]
 for name,g in df.groupby(scene):
  val=spearmanr(g[x],g[y]).statistic
  rows.append({'scene':name,'scene_number':scene_num(name),'srcc':float(val) if np.isfinite(val) else 0.,'constant':bool(not np.isfinite(val))})
 return pd.DataFrame(rows)
a['negative_lpips']=-a.lpips
cs=correlations(a,'filename','prediction','negative_lpips')
cs.sort_values('scene_number').to_csv(R/'lol_scene_srcc.csv',index=False,lineterminator='\r\n')
assert cs.constant.sum()==0,'Report constants before proceeding'
targets={'highest':cs.srcc.max(),'median':cs.srcc.median(),'lowest':cs.srcc.min()}
chosen=[]
for label,target in targets.items():
 tied=cs[np.isclose(cs.srcc,target,rtol=0,atol=1e-12)].sort_values('scene_number')
 row=tied.iloc[0].to_dict();row.update({'selection':label,'number_of_tied_scenes':len(tied)})
 data=a[a.filename==row['scene']].copy()
 row['lpips_best']=data.loc[np.isclose(data.lpips,data.lpips.min(),rtol=0,atol=1e-12),'method'].tolist()
 row['nr_free_best']=data.loc[np.isclose(data.prediction,data.prediction.max(),rtol=0,atol=1e-12),'method'].tolist()
 row['records']=data.to_dict('records');chosen.append(row)
b=pd.read_csv(P/'studies/records/squarelol_frozen_de_predictions.csv');b['name']=b.name.str.strip()
assert len(b)==2900 and b.scene.nunique()==290 and b.groupby('scene').size().eq(10).all()
cb=correlations(b,'scene','prediction','mos');assert cb.constant.sum()==0
cb.sort_values('scene_number').to_csv(R/'squarelol_scene_srcc.csv',index=False,lineterminator='\r\n')
median=cb.srcc.median();cb['distance_from_median']=abs(cb.srcc-median)
tied=cb[np.isclose(cb.distance_from_median,cb.distance_from_median.min(),rtol=0,atol=1e-12)].sort_values('scene_number')
scene=tied.iloc[0].to_dict();scene['population_median_srcc']=median;scene['number_of_tied_scenes']=len(tied)
g=b[b.scene==scene['scene']].copy();g['method']=g.name.str.replace(scene['scene']+'_','',regex=False).str.replace('_0.png','',regex=False)
g['human_rank']=g.mos.rank(ascending=False,method='average');g['nr_free_rank']=g.prediction.rank(ascending=False,method='average')
g=g.sort_values(['mos','method'],ascending=[False,True]);scene['records']=g.to_dict('records')
g.to_csv(R/'selected_squarelol_outputs.csv',index=False,lineterminator='\r\n')
a[a.filename.isin([x['scene'] for x in chosen])].to_csv(R/'selected_lol_outputs.csv',index=False,lineterminator='\r\n')
result={'plan_sha256':sha(plan_path),'selection_before_image_access':True,'OOF_validated_against_validation_predictions':True,'lol':chosen,'squarelol':scene,'constants_lol':int(cs.constant.sum()),'constants_squarelol':int(cb.constant.sum())}
dump(R/'selection.json',result)
print(json.dumps({**{x['selection']:{k:x[k] for k in ['scene','srcc','number_of_tied_scenes','lpips_best','nr_free_best']} for x in chosen},'squarelol':{k:scene[k] for k in ['scene','srcc','population_median_srcc','number_of_tied_scenes']}},indent=2))
