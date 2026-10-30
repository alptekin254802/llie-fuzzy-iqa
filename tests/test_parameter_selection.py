import unittest
import numpy as np
from analysis.parameter_selection.run import choose
from analysis.parameter_selection.verify import kernels
from analysis.parameter_selection.scorers import load_models
from evaluation.frozen_scorer import FrozenScorer

class ParameterSelection(unittest.TestCase):
    def test_fixed_tie_rule(self):
        p=[-1.,.5,1.,1.5,2.,3.,4.,6.,8.]
        self.assertEqual(choose(p,np.zeros(9),'clahe'),4)
        self.assertEqual(choose(p,[0,0,1,0,0,1,0,0,0],'clahe'),2)
    def test_release_models_unchanged(self):
        import pandas as pd
        x=np.random.default_rng(81).uniform([0,0,0],[8,.5,4],(50,3))
        df=pd.DataFrame(x,columns=['entropy','contrast','sharpness'])
        for public,key in [('nr_free','NR-free'),('nr_mono','NR-mono')]:
            np.testing.assert_allclose(load_models()[key].predict(df),FrozenScorer(public).predict(x),rtol=0,atol=1e-12)
    def test_exact_search_pixels_and_native_output(self):self.assertEqual(kernels()['native_parameter_checks'],9)
