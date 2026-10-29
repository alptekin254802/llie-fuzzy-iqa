"""Scene-held-out validation on the seven-enhancer LOL pool."""
from analysis.make_figures import load_data,orient,build_firing,calibrate,group_kfold,rho,NR_FEATURES,RESULTS
import numpy as np,json
from threadpoolctl import threadpool_limits
def main():
    threadpool_limits(1);_,d=load_data();y=-d.lpips.to_numpy();arr,_=orient(d,NR_FEATURES,y);F,den,_=build_firing(arr)
    pred=np.empty(len(d));fold=np.empty(len(d),int);rows={k:[] for k in ('nr_free','psnr','ssim')}
    for i,(tr,te) in enumerate(group_kfold(d.filename,5,42)):
        th=calibrate(F,den,y,tr,0.);pred[te]=F[te]@th/den[te];fold[te]=i
        for name,values in [('nr_free',pred),('psnr',d.psnr.to_numpy()),('ssim',d.ssim.to_numpy())]:rows[name].append(rho(values[te],y[te]))
    result={'images':len(d),'scenes':d.filename.nunique(),'protocol':'seven-method calibration; global orientations/partitions; train-fold consequents','models':{k:{'folds':v,'mean':float(np.mean(v)),'std':float(np.std(v))} for k,v in rows.items()}}
    (RESULTS/'expanded_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    d[['filename','method']].assign(fold=fold,prediction=pred).to_csv(RESULTS/'expanded_validation_predictions.csv',index=False)
    print(result)
if __name__=='__main__':main()
