import unittest
import numpy as np
from analysis.studies.common import within_group
from analysis.studies.human_objectives import within_srcc
class RankProtocolTests(unittest.TestCase):
    def test_tied_predictions_receive_half_credit_and_uniform_top_choice(self):
        r=within_group([3,3,1],[3,2,1],['a']*3)
        self.assertAlmostEqual(r['pair_acc'],5/6)
        self.assertAlmostEqual(r['top1'],.5)
    def test_human_summary_uses_average_ranks_and_zero_for_constant_scenes(self):
        score=np.array([3.,3.,1.,4.,4.,4.]);target=np.array([3.,2.,1.,3.,2.,1.])
        scene=np.array([[0,1,2],[3,4,5]])
        self.assertAlmostEqual(within_srcc(score,target,scene),np.sqrt(3)/4)
if __name__=='__main__':unittest.main()
