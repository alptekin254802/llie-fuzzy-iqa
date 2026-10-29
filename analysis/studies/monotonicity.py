"""Constrained minimum versus product firing; grid includes flat outer shoulders."""
from analysis.studies.support import *
from analysis.monotone_fuzzy import design as product_design
from evaluation.frozen_scorer import FrozenScorer
def main():
    data=c.lol_table();y=-data.lpips.to_numpy();sg,mfs,_=c.lol_frozen_model();arr=np.column_stack(c.oriented(data,sg))
    f,den=c.firing([arr[:,n] for n in range(3)],mfs);th,certificate=mono_fit(f/den[:,None],y)
    axes=[np.linspace(a.min()-.1*np.ptp(a),a.max()+.1*np.ptp(a),31) for a in arr.T]
    grid=np.array(list(itertools.product(*axes)));G,den=c.firing([grid[:,n] for n in range(3)],mfs)
    scores={'ordered_minimum':G@th/den,'nr_mono':product_design(grid,FrozenScorer('nr_mono').mfs)@FrozenScorer('nr_mono').theta}
    result={'grid_size_per_axis':31,'tolerance':1e-9,'ordered_minimum_certificate':certificate,'models':{}}
    for key,pred in scores.items():
        cube=pred.reshape(31,31,31)
        result['models'][key]={'decreasing_edges':{c.NR[n]:int(np.sum(np.diff(cube,axis=n)<-1e-9)) for n in range(3)},'largest_drop':float(max(0.,max(-np.diff(cube,axis=n).min() for n in range(3))))}
    save(OUT/'monotonicity.json',result)
if __name__=='__main__':main()
