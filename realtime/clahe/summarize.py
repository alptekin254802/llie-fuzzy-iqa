from pathlib import Path
import argparse,datetime,json
import numpy as np,pandas as pd
from analysis.parameter_selection.common import sha,save,csv as writecsv,output_dir,REPO
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stats(v):
 v=np.asarray(v,float)
 return dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),max_ms=float(v.max()),mean_ms=float(v.mean()),nominal_exceedances=int((v>1000/30).sum()))
def main():
 rows=[];raws=[]
 for n in [1,2,3]:
  folder=R/'offline'/f'session{n}'
  if not (folder/'status.json').exists():continue
  meta=json.loads((folder/'status.json').read_text())
  if meta['status']!='COMPLETE':continue
  assert sha(folder/'frames.csv')==meta['raw_sha256']
  z=pd.read_csv(folder/'frames.csv');assert len(z)==900;raws.append(z)
  for k,g in z.groupby('kind'):rows.append(dict(session=str(n),kind=k,**stats(g.milliseconds)))
 if len(raws)==3:
  z=pd.concat(raws);p=z.pivot(index=['session','repeat','scene'],columns='kind',values='param');assert (p.selection256==p.native).all()
  for k,g in z.groupby('kind'):rows.append(dict(session='pooled',kind=k,**stats(g.milliseconds)))
 writecsv(OUT/'offline/summary.csv',rows)
 live=[];tables={}
 for folder in sorted((R/'live').glob('round*_*')):
  if not (folder/'status.json').exists():continue
  meta=json.loads((folder/'status.json').read_text())
  if meta['status']!='COMPLETE':continue
  assert sha(folder/'frames.csv')==meta['raw_sha256']
  z=pd.read_csv(folder/'frames.csv');assert len(z)==2000
  assert z.warmup.astype(bool).equals(pd.Series(np.arange(2000)<50))
  z=z.loc[~z.warmup].copy();assert len(z)==1950
  for name,a,b in [('capture_ms','capture_end_ns','capture_start_ns'),('processing_ms','output_end_ns','capture_end_ns'),('capture_to_output_ms','output_end_ns','capture_start_ns')]:assert np.allclose(z[name],(z[a]-z[b])/1e6,rtol=0,atol=1e-10)
  tables[(meta['round'],meta['kind'])]=z
 def item(number,kind,z):
  return dict(session=str(number),kind=kind,**stats(z.processing_ms),loop_fps=float(len(z)/z.end_to_end_loop_ms.sum()*1000),capture_mean_ms=float(z.capture_ms.mean()),output_p95_ms=float(z.end_to_end_loop_ms.quantile(.95)),historical_misses=int((z.processing_ms>z.camera_interval_ms).sum()),read_intervals_over48=int((z.camera_interval_ms>48).sum()))
 for (number,kind),z in tables.items():live.append(item(number,kind,z))
 for kind in ['capture','gamma','clahe','clahe2']:
  sets=[z for (n,k),z in tables.items() if k==kind]
  if len(sets)==3:live.append(item('pooled',kind,pd.concat(sets)))
 if live:writecsv(OUT/'live/summary.csv',live)
 criteria=[]
 for n in [1,2,3]:
  for kind in ['clahe','clahe2']:
   a=next((r for r in live if r['session']==str(n) and r['kind']==kind),None);b=next((r for r in live if r['session']==str(n) and r['kind']=='capture'),None)
   if a and b:criteria.append(dict(session=n,kind=kind,relative_fps=a['loop_fps']/b['loop_fps'],near30=a['loop_fps']>=29 and a['loop_fps']>=.98*b['loop_fps'],p95_under_budget=a['p95_ms']<1000/30))
 save(OUT/'RESULTS.json',dict(utc=now(),offline=rows,live=live,criteria=criteria,serial_live_complete=sum(x['kind']=='clahe' for x in criteria)==3,scope='native output timing, no native quality test; no sensor timestamps or true dropped-frame claim'))
 print(json.dumps({'offline':rows,'live':live,'criteria':criteria}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=REPO/'results/studies/clahe_live');p.add_argument('--output',required=True);a=p.parse_args();R=a.input;OUT=output_dir(a.output);main()
