"""Scene bootstrap of pretrained-metric agreement with human scores, seed 7."""
from analysis.studies.support import *
LOWER={'niqe','brisque','lpips','lpips-vgg','dists'}
def main():
    result={};diagnostic={};signs,mfs,th=c.lol_frozen_model()
    for ds in ('squarelol','lieq','rlie'):
        scores=pd.read_csv(OUT/f'human_metrics_{ds}.csv')
        features=pd.read_csv(OUT/f'feat_{ds}_area256.csv')
        x=scores.merge(features[['name','scene','method','mos']+c.NR],on='name',validate='one_to_one')
        frozen=c.fuzzy_predict(x,signs,mfs,th);base=scene_values(frozen,x.mos,x.scene);rng=np.random.default_rng(7)
        samples=[rng.integers(len(base),size=len(base)) for _ in range(B)];result[ds]={}
        for metric in scores.columns:
            if metric=='name' or x[metric].isna().any():continue
            pred=x[metric].to_numpy()*(-1 if metric in LOWER else 1);v=scene_values(pred,x.mos,x.scene);draw=np.array([np.mean(v[i]-base[i]) for i in samples])
            result[ds][metric]={'within_scene_srcc':float(v.mean()),'difference':float((v-base).mean()),'bootstrap_mean_difference':float(draw.mean()),'ci95':np.quantile(draw,[.025,.975]).tolist(),'replicates':B,'seed':7}
        if ds!='rlie':
            counts={}
            for metric in ('lpips','mos'):
                counts[metric]={}
                for _,z in x.groupby('scene'):
                    vals=-z.lpips if metric=='lpips' else z.mos;best=z.loc[vals==vals.max(),'method']
                    for m in best:counts[metric][m]=counts[metric].get(m,0.)+1/len(best)
            diagnostic[ds]={'scenes':int(x.scene.nunique()),'uniform_top_method_counts':counts}
    save(OUT/'human_metric_uncertainty.json',result);save(OUT/'human_anchor_diagnostics.json',diagnostic)
if __name__=='__main__':main()
