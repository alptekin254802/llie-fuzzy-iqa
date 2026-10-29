"""Prepare keyed validation inputs without copying any raw image or human rating."""
import json, shutil
from analysis.studies import common as c
def main():
    primary=c.pd.read_csv(c.RES/'validation_predictions.csv')
    features=c.lol_table()
    needed=[k for k in features if k not in primary or k in ('filename','method')]
    merged=primary.merge(features[needed],on=['filename','method'],validate='one_to_one')
    assert len(merged)==2425 and merged.groupby('filename').fold.nunique().max()==1
    path=c.OUT/'validation_inputs.csv'
    if path.exists():
        old=c.pd.read_csv(path)
        assert old.shape==merged.shape and list(old)==list(merged), 'Existing input differs'
        c.np.testing.assert_allclose(old.select_dtypes('number'),merged.select_dtypes('number'),rtol=0,atol=1e-12)
    else:merged.to_csv(path,index=False)
    print('Prepared',len(merged),'validation rows at',path)
if __name__=='__main__':main()
