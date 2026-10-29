"""Run method-coverage reproduction steps in a separate configured directory."""
import argparse,importlib,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['prepare','run','analyze','verify','summary'])
    p.add_argument('--config',type=Path,help='Study data/cache configuration')
    p.add_argument('--out',type=Path,help='Separate reproduction output directory')
    a=p.parse_args()
    if a.config:os.environ['LLIE_STUDY_CONFIG']=str(a.config.resolve())
    if a.out:os.environ['LLIE_METHOD_COVERAGE_WORK']=str(a.out.resolve())
    if a.step=='summary':
        import pandas as pd,numpy as np
        root=Path(__file__).resolve().parents[2]/'results/studies/method_coverage'
        results=pd.read_csv(root/'results.csv');curve=pd.read_csv(root/'curve_summary.csv');contrasts=pd.read_csv(root/'contrasts.csv')
        assert len(results)==4500 and results.status.eq('COMPLETE').all()
        for r in curve.itertuples():
            x=results[(results.dataset==r.dataset)&(results.model==r.model)&(results.k==r.k)].SRCC
            assert len(x)==100 and np.isclose(x.mean(),r.mean_SRCC,rtol=0,atol=1e-12) and np.isclose(x.std(ddof=0),r.SD,rtol=0,atol=1e-12)
        for r in contrasts.itertuples():
            v=curve[(curve.dataset==r.dataset)&(curve.model==r.model)&(curve.k==r.k)].mean_SRCC.iloc[0]
            w=curve[(curve.dataset==r.dataset)&(curve.model==r.reference_model)&(curve.k==r.reference_k)].mean_SRCC.iloc[0]
            assert np.isclose(v-w,r.difference,rtol=0,atol=1e-12)
        print('PASS: 4500 score conditions, 45 curve summaries, 33 contrast point estimates. Interval reproduction requires the full pipeline.')
    else:importlib.import_module('analysis.method_coverage.'+a.step).main()
if __name__=='__main__':main()
