"""Apply the included frozen models to human-rated images without fitting."""
from analysis.studies.support import *
from evaluation.frozen_scorer import FrozenScorer
from analysis.monotone_fuzzy import design as product_design
def main():
    result={};lol=c.lol_table();low=lol[c.NR].min();high=lol[c.NR].max()
    for ds in ('squarelol','lieq','rlie'):
        x=pd.read_csv(OUT/f'feat_{ds}_area256.csv',dtype={'scene':str},float_precision='round_trip');result[ds]={}
        for key in ('nr_free','nr_mono'):
            model=FrozenScorer(key)
            if key=='nr_mono':p=product_design(x[c.NR].to_numpy()*model.signs,model.mfs)@model.theta
            else:p=c.fuzzy_predict(x,model.parameters['signs'],model.mfs,model.theta)
            result[ds][key]=summary(p,x.mos,x.scene)
            x[['name','scene']].assign(prediction=p).to_csv(OUT/f'{key}_{ds}_predictions.csv',index=False)
        outside=((x[c.NR]<low)|(x[c.NR]>high))
        result[ds]['outside_lol_range']={'n':len(x),'any_fraction':float(outside.any(axis=1).mean()),'by_feature':outside.mean().to_dict()}
    save(OUT/'canonical_human.json',result)
if __name__=='__main__':main()
