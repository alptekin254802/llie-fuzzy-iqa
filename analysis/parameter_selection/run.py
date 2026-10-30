"""Regenerate candidate features, selections and full-reference quality from local images."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,csv as csv_module,itertools,json
import numpy as np
import pandas as pd
import cv2
from threadpoolctl import threadpool_limits
from .common import DATA,configure,read,csv,save,output_dir,sha
from .candidates import candidates,read_input,read_reference,features,compute_psnr,compute_ssim,FAMILIES
from .scorers import score_models,scope_for_scene,NAMES

DEFAULTS={'clahe':2.,'log':16.,'he_blend':1.,'denoise':0.}
KEY=['dataset','scene','candidate_id']

def choose(parameters,scores,family):
    scores=np.asarray(scores,float);p=np.asarray(parameters,float)
    if not np.isfinite(scores).all():raise ValueError('Non-finite score')
    eligible=np.flatnonzero(scores>=scores.max()-1e-9)
    return min(eligible,key=lambda i:(round(abs(p[i]-DEFAULTS[family]),12),p[i]))

def decisions(frame,scope,confirmation=False):
    controls=json.loads((DATA/'controls.json').read_text())[scope]['families'];out=[]
    families=['clahe'] if confirmation else list(FAMILIES)
    for family in families:
        z=frame[frame.family==family].sort_values('param').reset_index(drop=True)
        def add(index,name):
            row=z.iloc[index].to_dict();out.append(dict(**{k:row[k] for k in KEY+['reference_group','family','param']},selector=name))
        for name in [*NAMES,*(['RF3','RF4'] if confirmation else []),'entropy']:
            add(choose(z.param,z[name],family),name)
        for name,target in [('brightness_045',.45),('brightness_tuned',controls[family]['brightness_target'])]:
            add(choose(z.param,-abs(z.brightness-target),family),name)
        for name,value in [('training_fixed',controls[family]['fixed_param']),('default',DEFAULTS[family])]:
            add(int(np.flatnonzero(z.param==value)[0]),name)
        raw=frame[frame.candidate_id==('clahe:-1' if confirmation else 'control:rawinput')].iloc[0]
        out.append(dict(**{k:raw[k] for k in KEY+['reference_group','param']},family=family,selector='raw_input'))
        if family=='log':
            row=frame[frame.candidate_id=='control:original_log'].iloc[0]
            out.append(dict(**{k:row[k] for k in KEY+['reference_group','param']},family=family,selector='original_log'))
    return pd.DataFrame(out)

def append(path,frame):
    frame.to_csv(path,index=False,mode='a',header=not path.exists(),lineterminator='\r\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['screen','confirmation'],required=True);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    configure(a.config);out=output_dir(a.output);threadpool_limits(1);cv2.setNumThreads(1)
    manifest=read(DATA/a.part/'inputs.csv').fillna('');confirm=a.part=='confirmation'
    freeze={'part':a.part,'inputs_sha256':sha(DATA/a.part/'inputs.csv'),'models_sha256':sha(DATA/'models.json'),'controls_sha256':sha(DATA/'controls.json'),'scopes_sha256':sha(DATA/'scopes.json'),'model_parameters':{p.relative_to(DATA).as_posix():sha(p) for p in (DATA/'models').rglob('*.json')}}
    lock=out/'INPUTS.json'
    if lock.exists():
        if json.loads(lock.read_text())!=freeze:raise ValueError('Inputs changed; use a new output directory')
    else:save(lock,freeze)
    # A completed image is the resume unit. Partial image files are rejected.
    done=set();present=[(out/n).exists() for n in ['candidates.csv','selected_quality.csv']]
    if any(present) and not all(present):raise ValueError('Partial image write; preserve files and recover before resuming')
    for filename in ['candidates.csv','selected_quality.csv']:
        p=out/filename
        if p.exists():
            z=read(p);seen=set(map(tuple,z[['dataset','scene']].drop_duplicates().to_numpy()))
            expected=(9 if confirm else 35) if filename=='candidates.csv' else (13 if confirm else 45)
            if not z.groupby(['dataset','scene']).size().eq(expected).all():raise ValueError('Partial image rows; preserve files and recover before resuming')
            if done and seen!=done:raise ValueError('Partial image write; preserve files and recover before resuming')
            done=seen
    import torch,pyiqa
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    metrics={k:pyiqa.create_metric(n,device=a.device) for k,n in [('lpips','lpips'),('lpips_vgg','lpips-vgg'),('dists','dists')]}
    for row in manifest.to_dict('records'):
        if (row['dataset'],row['scene']) in done:continue
        low=read_input(row);reference=read_reference(row)
        cand=list(itertools.islice(candidates(low),9)) if confirm else list(candidates(low))
        records=[dict(dataset=row['dataset'],scene=row['scene'],reference_group=row['reference_group'],family=fam,param=param,candidate_id=cid,**features(im)) for fam,param,cid,im in cand]
        frame=pd.DataFrame(records);scope=scope_for_scene(row['scene']) if row['dataset']=='LOL' else 'final'
        scores=score_models(frame,scope,with_forests=confirm)
        for name in scores:frame[name]=scores[name].to_numpy()
        selected=decisions(frame,scope,confirm)
        # Freeze image-only choices before reading any candidate quality values.
        quality=[]
        with torch.inference_mode():
            ref=torch.from_numpy(reference.copy()).permute(2,0,1)[None].to(a.device)
            for i,(_,_,_,im) in enumerate(cand):
                x=torch.from_numpy(im.copy()).permute(2,0,1)[None].to(a.device)
                vals={k:float(m(x,ref).item()) for k,m in metrics.items()}
                quality.append(dict(**{k:records[i][k] for k in KEY},**vals,psnr=float(compute_psnr(im,reference)),ssim=float(compute_ssim(im,reference))))
        q=pd.DataFrame(quality)
        append(out/'candidates.csv',frame.merge(q,on=KEY,validate='one_to_one'))
        append(out/'selected_quality.csv',selected.merge(q,on=KEY,validate='many_to_one'))
        print(row['dataset'],row['scene'],flush=True)
    save(out/'COMPLETE.json',{'images':len(manifest),'part':a.part,'candidate_sha256':sha(out/'candidates.csv'),'selected_quality_sha256':sha(out/'selected_quality.csv'),'note':'Oracle choices and grouped analysis are separate from image-only selection.'})
if __name__=='__main__':main()
