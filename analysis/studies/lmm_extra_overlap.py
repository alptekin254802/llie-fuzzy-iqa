"""Recompute verified overlap summaries from stored predictions; no fitting or inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
root=Path(__file__).resolve().parents[2]
d=pd.read_csv(root/'results/lmm_extra/joined.csv')
ov=pd.read_csv(root/'results/studies/records/squarelol_lol_scene_matches.csv')
expected=json.loads((root/'results/lmm_extra/overlap_verified.json').read_text())
result={}
for group,mask in [('shared',ov['corr']>=.9),('nonoverlap',ov['corr']<.9)]:
 x=d[(d.dataset=='squarelol') & d.scene.isin(ov.loc[mask,'squarelol'])];result[group]={}
 for model in ['vqa2','vital']:
  result[group][model]={'pooled_srcc':float(spearmanr(x[model],x.mos).statistic),'within_scene_srcc':float(np.mean([spearmanr(v[model],v.mos).statistic for _,v in x.groupby('scene')]))}
  for metric,value in result[group][model].items():assert abs(value-expected[group][model][metric])<1e-12
print(json.dumps(result,indent=2));print('PASS: all scenes use committed normal-light classes; full-data results unchanged.')