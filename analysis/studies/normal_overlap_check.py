"""Validate normal-light overlap classification before using it for a missing low-light source."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import pandas as pd
import cv2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\r\n')
def thumb(p):
 g=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2GRAY)
 x=cv2.equalizeHist(cv2.resize(g,(48,32),interpolation=cv2.INTER_AREA)).astype(np.float64).ravel()
 return (x-x.mean())/(x.std()+1e-9)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--square-normal-dir',type=Path,required=True);ap.add_argument('--lol-normal-dir',type=Path,required=True);ap.add_argument('--matches',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'PLAN.json').exists(),'Choose a new audit output; do not overwrite the preregistration.'
 labels=pd.read_csv(args.matches);known=labels[labels['corr'].notna()].copy();missing=labels[labels['corr'].isna()].copy();assert len(known)==289 and missing.squarelol.tolist()==['low00710']
 lo=sorted(args.lol_normal_dir.glob('*.png'));assert len(lo)==485
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'known_scenes':len(known),'missing_source_scenes':missing.squarelol.tolist(),'label_record_sha256':sha(args.matches),'thumbnail':'BGR to grayscale; area 48x32; equalizeHist; float64 centering and division by std+1e-9; mean dot product','threshold':.9,'normal_source_mapping':'lowNNNNN -> normalNNNNN.png','validation':'Compare normal-light shared/non-overlap classification to the recorded low-light classification on all 289 known scenes.','gate':'Only exact 289/289 class agreement permits classifying low00710 with its normal-light reference. Otherwise retain unknown; do not tune threshold or preprocessing.','score_retraining':False}
 dump(out/'PLAN.json',plan);(out/'PLAN.sha256').write_text(sha(out/'PLAN.json')+'\n',encoding='utf-8',newline='\r\n')
 B=np.stack([thumb(p) for p in lo]);rows=[];hashes={'lol':{p.name:sha(p) for p in lo},'squarelol':{}}
 def classify(scene):
  p=args.square_normal_dir/(scene.replace('low','normal',1)+'.png');assert p.exists(),p
  x=thumb(p);v=B@x/x.size;j=int(v.argmax());hashes['squarelol'][p.name]=sha(p)
  return {'squarelol':scene,'normal_file':p.name,'best_lol_normal':lo[j].name,'normal_corr':float(v[j]),'normal_shared':bool(v[j]>=plan['threshold'])}
 for r in known.itertuples(index=False):
  row=classify(r.squarelol);row.update(lowlight_corr=float(r.corr),lowlight_shared=bool(r.shared_source),lowlight_best_lol=r.best_our485);row['class_agrees']=row['normal_shared']==row['lowlight_shared'];rows.append(row)
 df=pd.DataFrame(rows);df.to_csv(out/'known_scene_validation.csv',index=False,lineterminator='\r\n');n=int(df.class_agrees.sum());passed=n==len(df)
 target=classify('low00710') if passed else {'squarelol':'low00710','classification':'unknown','normal_test_applied':False,'reason':'Known-scene agreement was not complete.'}
 if passed:target.update(classification='shared' if target['normal_shared'] else 'nonoverlap',normal_test_applied=True,method='normal-light thumbnail check; low-light source missing')
 counts={'shared':int(known.shared_source.eq(True).sum()),'nonoverlap':int(known.shared_source.eq(False).sum()),'unknown':len(missing)}
 if passed:counts[target['classification']]+=1;counts['unknown']-=1
 result={'plan_sha256':sha(out/'PLAN.json'),'validation_n':len(df),'agreement_n':n,'agreement_fraction':n/len(df),'agreement_percent':100*n/len(df),'perfect_agreement':passed,'mismatches':df[~df.class_agrees].to_dict('records'),'normal_shared_in_validation':int(df.normal_shared.sum()),'lowlight_shared_in_validation':int(df.lowlight_shared.sum()),'candidate':target,'final_counts':counts,'threshold':plan['threshold'],'normal_highest_match_id_agreement_n':int((df.best_lol_normal==df.lowlight_best_lol).sum()),'source_hashes':hashes}
 dump(out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()