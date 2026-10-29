"""Recompute all recorded hand-tuned baseline scores from their metric inputs."""
from pathlib import Path
import pandas as pd,numpy as np,json
from evaluation.handtuned_baseline import FuzzyEnhancementEvaluator
def main():
    root=Path(__file__).resolve().parents[1];d=pd.read_csv(root/'results/fuzzy_enhancement_results.csv');model=FuzzyEnhancementEvaluator()
    pred=np.array([model.evaluate(float(r.psnr),float(r.ssim),float(r.entropy))[0] for _,r in d.iterrows()])
    result={'rows':len(d),'maximum_absolute_difference':float(np.max(abs(pred-d.fuzzy_score.to_numpy())))}
    (root/'results/studies/handtuned_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['maximum_absolute_difference']>1e-6:raise RuntimeError('Baseline reconstruction differs from recorded scores')
    print(result)
if __name__=='__main__':main()
