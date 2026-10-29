"""Identify unsupported antecedents and test their bounded consequent sensitivity."""
from analysis.studies.support import *
from evaluation.frozen_scorer import FrozenScorer
from analysis.monotone_fuzzy import design as product_design
def main():
    lol=c.lol_table();human=external('squarelol');out={}
    for name in ('nr_free','nr_mono'):
        model=FrozenScorer(name);sg=model.parameters['signs'];mfs=model.mfs
        if name=='nr_free':F,den=c.firing(c.oriented(lol,sg),mfs);F=F/den[:,None];H,den=c.firing(c.oriented(human,sg),mfs);H=H/den[:,None]
        else:F=product_design(lol[c.NR].to_numpy()*model.signs,mfs);H=product_design(human[c.NR].to_numpy()*model.signs,mfs)
        unused=np.flatnonzero(F.max(axis=0)==0)
        base=within(H@model.theta,human.mos,human.scene)['mean_srcc'];tests={}
        if name=='nr_free':
            for i in unused:
                tests[str(i+1)]={}
                for value in (0.,100.):
                    th=model.theta.copy();th[i]=value
                    tests[str(i+1)][str(value)]=within(H@th,human.mos,human.scene)['mean_srcc']-base
        out[name]={'zero_activation_rules_one_based':(unused+1).tolist(),'max_training_activation':F.max(axis=0).tolist(),'squarelol_within':base,'consequent_sensitivity':tests}
    save(OUT/'rule_support.json',out)
if __name__=='__main__':main()
