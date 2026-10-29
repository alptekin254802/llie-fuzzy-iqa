"""Verify every stored VQA2/VITAL score from five logits, without model loading."""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.stats import pearsonr,spearmanr
from .scoring import quality
ROOT=Path(__file__).resolve().parents[2]/'results/lmm_extra'
def rows(path):return list(csv.DictReader(path.open(encoding='utf-8',newline='')))
def verify_pilot(model,reference_path,candidate_path):
    reference=rows(reference_path);candidate=rows(candidate_path)
    expected=rows(ROOT/model/'pilot_reference.csv')[:20]
    keys=lambda data:[(r['dataset'],r['name'],r['rgb256_sha256']) for r in data]
    if len(reference)!=20 or len(candidate)!=20 or keys(reference)!=keys(expected) or keys(candidate)!=keys(expected):
        raise ValueError('Both pilot files must contain the same 20 recorded inputs in order')
    for data,precision in [(reference,'bf16_cpu'),(candidate,'nf4_fp32')]:
        for row in data:
            if row['precision']!=precision or abs(quality(json.loads(row['level_logits_json']))-float(row['score']))>1e-12:
                raise ValueError('Pilot precision or score/logit mismatch')
    joined={(r['dataset'],r['name']):r for r in rows(ROOT/'joined.csv')}
    a=[float(r['score']) for r in candidate];b=[float(r['score']) for r in reference]
    q=[float(joined[(r['dataset'],r['name'])]['qalign']) for r in candidate]
    srcc=float(spearmanr(a,b).statistic);qp=float(pearsonr(a,q).statistic);qs=float(spearmanr(a,q).statistic)
    import hashlib
    return {'model':model,'precision':'nf4_fp32','accepted':bool(np.isfinite([srcc,qp,qs]).all() and srcc>=.98 and qp>0 and qs>0),
            'reference_srcc':srcc,'qalign_pearson':qp,'qalign_srcc':qs,
            'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest()}

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['vqa2','vital']);parser.add_argument('--reference',type=Path)
    parser.add_argument('--candidate',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if any([args.model,args.reference,args.candidate,args.output]):
        if not all([args.model,args.reference,args.candidate,args.output]):parser.error('Provide model, reference, candidate and output together')
        output=args.output.resolve();repo=ROOT.parents[1]
        if repo in output.parents and repo/'study_work' not in output.parents:parser.error('Write local pilot decisions under study_work or outside the repository')
        report=verify_pilot(args.model,args.reference,args.candidate)
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
        if not report['accepted']:raise SystemExit('Pilot concordance/direction rule failed')
        return
    joined={(r['dataset'],r['name']):r for r in rows(ROOT/'joined.csv')}
    manifest=rows(ROOT/'image_manifest.csv');result={}
    for model in ['vqa2','vital']:
        scores=rows(ROOT/model/'scores.csv')
        assert len(scores)==len(manifest)==5440
        error=0
        for row,expected in zip(scores,manifest):
            key=row['dataset'],row['name']
            assert all(row[k]==expected[k] for k in ['dataset','name','scene','rgb256_sha256'])
            value=quality(json.loads(row['level_logits_json']))
            error=max(error,abs(value-float(row['score'])))
            assert math.isclose(value,float(row['score']),abs_tol=1e-12,rel_tol=0)
            assert math.isclose(value,float(joined[key][model]),abs_tol=1e-12,rel_tol=0)
        reference=rows(ROOT/model/'pilot_reference.csv');candidate=rows(ROOT/model/'pilot_candidate.csv')
        assert len(reference)>=20 and len(candidate)==20
        assert [(r['dataset'],r['name']) for r in reference[:20]]==[(r['dataset'],r['name']) for r in candidate]
        for row in reference+candidate:assert abs(quality(json.loads(row['level_logits_json']))-float(row['score']))<1e-12
        a=[float(r['score']) for r in candidate];b=[float(r['score']) for r in reference[:20]]
        corr=float(spearmanr(a,b).statistic);assert corr>=.98
        q=[float(joined[(r['dataset'],r['name'])]['qalign']) for r in candidate]
        assert pearsonr(a,q).statistic>0 and spearmanr(a,q).statistic>0
        result[model]={'scores':len(scores),'max_logit_reconstruction_error':error,'pilot_n':20,'pilot_srcc':corr,'pilot_pearson':float(pearsonr(a,b).statistic),'median_inference_seconds':float(np.median([float(r['inference_ms'])/1000 for r in scores]))}
    print(json.dumps({'status':'PASS','models':result},indent=2))
if __name__=='__main__':main()
