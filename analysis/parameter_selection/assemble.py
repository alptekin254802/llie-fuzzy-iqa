"""Turn regenerated candidate/selection files into the reported grouped tables."""
import argparse
import numpy as np,pandas as pd
from .common import read,csv,output_dir
from .run import KEY,FAMILIES,DEFAULTS,choose
from .analyse import calculate

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['screen','confirmation'],required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from pathlib import Path
    source=Path(a.input);out=output_dir(a.output)
    candidates=read(source/'candidates.csv');selected=read(source/'selected_quality.csv')
    metric_names=['lpips','lpips_vgg','dists','psnr','ssim'];extra=[]
    selected_keys=set(map(tuple,selected[KEY].to_numpy()))
    for (ds,fam),z in candidates[candidates.family.isin(FAMILIES)].groupby(['dataset','family'],sort=True):
        fixed=min((g.groupby('reference_group').lpips.mean().mean(),abs(param-DEFAULTS[fam]),param) for param,g in z.groupby('param'))[-1]
        per_image=[]
        for _,g in z.groupby('scene',sort=True):
            g=g.sort_values('param').reset_index(drop=True);per_image.append(g.iloc[choose(g.param,-g.lpips,fam)])
        for name,ch in [('test_best_fixed_oracle',z[z.param==fixed]),('per_image_oracle',pd.DataFrame(per_image))]:
            ch=ch[KEY+['reference_group','family','param']+metric_names].copy();ch['selector']=name
            if a.part=='screen':
                # Secondary metrics originally covered the selected-output union.
                absent=[tuple(row) not in selected_keys for row in ch[KEY].to_numpy()]
                ch.loc[absent,['lpips_vgg','dists']]=np.nan
            extra.append(ch)
    selected=pd.concat([selected,*extra],ignore_index=True)
    csv(out/'selected_quality.csv',selected)
    grouped=selected.groupby(['dataset','family','reference_group','selector'],sort=True)[metric_names].mean()
    wide=grouped.stack(future_stack=True).rename('score').reset_index().rename(columns={'level_4':'metric'})
    wide=wide.pivot(index=['dataset','family','reference_group','metric'],columns='selector',values='score').reset_index()
    csv(out/'group_scores.csv',wide)
    tables=calculate(a.part,out/'group_scores.csv')
    for name,table in tables.items():csv(out/(name+'.csv'),table)
    print('Wrote grouped estimates and paired comparisons to',out)
if __name__=='__main__':main()
