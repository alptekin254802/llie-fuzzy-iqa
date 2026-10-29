"""Classify all SQUARE-LOL sources using normal-light images, before reading model scores."""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np
import pandas as pd
from analysis.studies.normal_overlap_check import thumb

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s,encoding='utf-8',newline='\r\n')
def dump(p,x): write(p,json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser()
 for n in ['square-normal-dir','lol-normal-dir','lowlight-labels','output','classification-record']: ap.add_argument('--'+n,type=Path,required=True)
 a=ap.parse_args(); o=a.output; o.mkdir(parents=True,exist_ok=True)
 assert not (o/'PLAN.json').exists(),'Use a new output directory.'
 labels=pd.read_csv(a.lowlight_labels); assert len(labels)==290 and labels.squarelol.nunique()==290
 images=sorted(a.lol_normal_dir.glob('*.png')); assert len(images)==485
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':'Use normal-light image matching for every rated scene; classify shared if maximum correlation >= 0.9, otherwise non-overlapping.','reason':'The preceding perfect-agreement gate incorrectly treated low-light matching as ground truth. On known scenes normal-light correlations separated the classes, while low00182 matched normal-light LOL 186.png at approximately 0.9997; the low-light search instead selected adjacent frame 185.png at approximately 0.873. Use image evidence, not model outcomes.','scope':'All 290 scenes, including low00182 and low00710.','preprocessing':'BGR to grayscale; 48x32 area interpolation; equalizeHist; float64; center and divide by std+1e-9; mean dot product.','threshold':0.9,'model_score_access':'No model score files are opened by this script. Commit classification and plan before recomputing subgroup metrics.','prior_lowlight_labels_sha256':sha(a.lowlight_labels),'prior_gate':'Replaced by the normal-light classification protocol based on image evidence; the complete earlier record is retained.','script_sha256':sha(Path(__file__))}
 dump(o/'PLAN.json',plan); write(o/'PLAN.sha256',sha(o/'PLAN.json')+'\n')
 write(o/'prior_lowlight_labels.csv',a.lowlight_labels.read_text(encoding='utf-8'))
 B=np.stack([thumb(p) for p in images]); rows=[]; hashes={'lol':{p.name:sha(p) for p in images},'squarelol':{}}
 for r in labels.itertuples(index=False):
  f=a.square_normal_dir/(r.squarelol.replace('low','normal',1)+'.png'); x=thumb(f); cor=B@x/x.size; j=int(cor.argmax()); value=float(cor[j]); hashes['squarelol'][f.name]=sha(f)
  rows.append({'squarelol':r.squarelol,'best_our485':images[j].name,'corr':value,'shared_source':value>=plan['threshold'],'status':'verified_normal_light_thumbnail','normal_file':f.name,'lowlight_best_our485':r.best_our485,'lowlight_corr':r.corr,'lowlight_shared_source':r.shared_source})
 df=pd.DataFrame(rows); df.to_csv(o/'classification.csv',index=False,lineterminator='\r\n'); df.to_csv(a.classification_record,index=False,lineterminator='\r\n')
 known=df[df.lowlight_corr.notna()]; changes=known[known.shared_source!=known.lowlight_shared_source]
 clean=lambda x:json.loads(x.to_json(orient='records'))
 result={'plan_sha256':sha(o/'PLAN.json'),'classification_sha256':sha(a.classification_record),'scene_count':len(df),'shared':int(df.shared_source.sum()),'nonoverlap':int((~df.shared_source).sum()),'known_lowlight_count':len(known),'lowlight_agreement':int((known.shared_source==known.lowlight_shared_source).sum()),'known_disagreements':clean(changes),'low00710':clean(df[df.squarelol.eq('low00710')])[0],'correlation_ranges':{'shared_min':float(df[df.shared_source]['corr'].min()),'nonoverlap_max':float(df[~df.shared_source]['corr'].max())},'image_sha256':hashes}
 dump(o/'results.json',result)
 write(o/'README.md','# Normal-light source classification\n\nAll rated scenes use the same normal-light thumbnail test and fixed threshold. Classification was committed before model subgroup metrics were recomputed. The earlier gate is preserved as history in the sibling normal_light_overlap_validation directory.\n\nRun from the repository root:\n\n```text\npython -m analysis.studies.normal_overlap_classify --square-normal-dir <SQUARE-LOL>/all_normal --lol-normal-dir <LOL>/our485/high --lowlight-labels results/studies/normal_light_overlap_validation/lowlight_labels.csv --output <new-audit-directory> --classification-record <new-classification.csv>\n```\n\nPLAN.json and its SHA-256 precede the image calculation. results.json contains the classification decision, image hashes and correlation separation; classification.csv contains all scene matches. No images or model outputs are included in this decision record.\n')
 print(json.dumps({k:v for k,v in result.items() if k!='image_sha256'},indent=2))
if __name__=='__main__':main()
