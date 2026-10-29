import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
from analysis.human_labels import attach_local_mos, local_output

class LocalLabelTests(unittest.TestCase):
    def test_join_by_name_not_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); names=np.empty((2,3),dtype=object); names[:]='unused'
            names[0,2]='enhancement\\002.png';names[1,2]='enhancement\\001.png'
            savemat(root/'imgNames.mat',{'imgNames':names});savemat(root/'MOS.mat',{'MOSz':np.array([2.,1.])})
            pd.DataFrame({'Image Name':['002.png','001.png'],'Score':[4.,3.]}).to_csv(root/'normalized_scores.csv',index=False)
            data=pd.DataFrame({'dataset':['lieq','rlie','squarelol','lieq'],'name':['001.png','002.png','one.png','002.png'],'mos':[np.nan,np.nan,7.,np.nan]})
            result=attach_local_mos(data,root,root)
            self.assertEqual(result.mos.tolist(),[1.,4.,7.,2.])
            self.assertTrue(data.mos.isna().iloc[0])
            bad=data.copy();bad.loc[0,'name']='missing.png'
            with self.assertRaisesRegex(ValueError,'Missing'):attach_local_mos(bad,root,root)
            bad=data.copy();bad.loc[0,'mos']=9
            with self.assertRaisesRegex(ValueError,'disagree'):attach_local_mos(bad,root,root)
            with self.assertRaisesRegex(ValueError,'Duplicate'):attach_local_mos(pd.concat([data,data.iloc[:1]]),root,root)
    def test_prevent_public_label_export(self):
        from analysis.human_labels import ROOT
        with self.assertRaisesRegex(ValueError,'ignored study_work'):local_output(ROOT/'results/lmm_extra/joined.csv')
if __name__=='__main__':unittest.main()
