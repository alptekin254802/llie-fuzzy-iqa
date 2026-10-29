"""Training-fold method means, compared with held-out model predictions."""
from analysis.studies.support import *
from analysis.studies.support_extra import within2
def main():
    d=pd.read_csv(OUT/'validation_inputs.csv'); y=-d.lpips.to_numpy(); folds=list(cv_folds(d)); p=np.empty(len(d)); means={}
    for i,(tr,te) in enumerate(folds):
        means[i]=pd.DataFrame({'method':d.method.iloc[tr].to_numpy(),'y':y[tr]}).groupby('method').y.mean().to_dict()
        p[te]=d.method.iloc[te].map(means[i])
    results={'training_method_means':means,'models':{},'within_method_note':'Within-method SRCC pools out-of-fold predictions. The method-mean baseline is constant within each held-out fold and method, but training means differ between folds; its pooled within-method SRCC is not a scene-ranking ability.'}
    predictions={'method_mean':p,'nr_free':d.oof_nr_free,'linear':d.oof_linear,'random_forest':d.oof_rf,'psnr':d.psnr,'ssim':d.ssim}
    for name,score in predictions.items():
        r=summary(score,y,d.filename,folds)
        r['within']=within2(score,y,d.filename)
        r['within_method']={m:rho(np.asarray(score)[idx],y[idx]) if np.unique(np.asarray(score)[idx]).size>1 else None for m,idx in d.groupby('method').indices.items()}
        results['models'][name]=r
    results['nr_free_minus_method_mean']=paired_bootstrap(d.oof_nr_free,p,y,d.filename,d.fold)
    results['nr_free_minus_method_mean']['within']=paired_within(d.oof_nr_free,p,y,d.filename)
    save(OUT/'method_mean.json',results)
    d[['filename','method','fold']].assign(prediction=p).to_csv(OUT/'method_mean_predictions.csv',index=False)
if __name__=='__main__':main()
