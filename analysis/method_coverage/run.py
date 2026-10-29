"""Run the 20 repeats, five scene folds and five training-method counts coverage design; record every model case."""
from .common import *
import logging,traceback

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'run.log',encoding='utf-8')])
    log=logging.getLogger('method_coverage');config=read('config.json')
    check(read('preflight.json')['status']=='PASS','S14 reproduction gate must pass')
    check(sha(ROOT/'config.json')==read('config_lock.json')['sha256'],'Fixed design changed')
    check(snapshot_inputs()==read('input_hashes.json'),'Scientific inputs changed')
    results=[];attempts=[];scene_parts=[];prediction_parts=[];start=time.perf_counter()
    save('run_status.json',dict(status='RUNNING',started_utc=now(),expected_score_cases=4500,expected_human_fits=3000,completed=0))
    signs,mfs,prior=c.lol_frozen_model()
    save('models/LOL_prior.json',dict(signs=signs,mfs=np.asarray(mfs).tolist(),theta=prior.tolist(),source='models/nr_free.json',fitted_human_labels=False))
    for ds in DATASETS:
        x,foldids=load(ds);D,X3,theta0=arrays(x);y=x.mos.to_numpy(float);meth=x.method.astype(str).to_numpy()
        ds_predictions=[];ds_scenes=[]
        with (ROOT/f'models/{ds}.jsonl').open('w',encoding='utf-8') as model_file:
            for plan in config['datasets'][ds]['splits']:
                rep=plan['repeat'];test_methods=plan['test_methods']
                for fold in range(5):
                    te=np.flatnonzero((foldids==fold)&np.isin(meth,test_methods))
                    xt=x.iloc[te].reset_index(drop=True);Ste=scene_matrix(xt)
                    check(Ste.shape[1]==4,'Test candidate set must contain4 methods')
                    scene_names=xt.scene.to_numpy()[Ste[:,0]]
                    for k in KS:
                        train_methods=plan['train_methods_by_k'][str(k)]
                        tr=np.flatnonzero((foldids!=fold)&np.isin(meth,train_methods))
                        check(not set(x.scene.iloc[tr])&set(x.scene.iloc[te]),'Scene leakage')
                        check(not set(train_methods)&set(test_methods),'Method leakage')
                        check(set(meth[tr])==set(train_methods) and set(meth[te])==set(test_methods),'Method row filtering mismatch')
                        Str=scene_matrix(x.iloc[tr].reset_index(drop=True))
                        check(Str.shape[1]==k,'Training candidate count')
                        yc=h.demean(y[tr],Str)
                        pred_table=xt[['name','scene','method','mos']].copy()
                        scene_table=pd.DataFrame(dict(dataset=ds,repeat=rep,k=k,fold=fold,scene=scene_names))
                        scene_table['target_constant']=np.ptp(y[te][Ste],axis=1)==0
                        for name in MODELS:
                            case_start=time.perf_counter();status='COMPLETE';error='';state={}
                            try:
                                if name=='LOL prior':pred=D[te]@theta0
                                elif name=='fuzzy LS':
                                    th=h.fit_ls_within(D[tr],y[tr],Str,ridge=1e-3)
                                    pred=D[te]@th
                                    state=dict(theta=th.tolist(),ridge=1e-3)
                                else:
                                    fit=make_pipeline(StandardScaler(),LinearRegression()).fit(h.demean(X3[tr],Str),yc)
                                    pred=fit.predict(h.demean(X3[te],Ste))
                                    state=dict(scaler_mean=fit[0].mean_.tolist(),scaler_scale=fit[0].scale_.tolist(),coef=fit[-1].coef_.tolist(),intercept=float(fit[-1].intercept_))
                                check(np.isfinite(pred).all(),'Nonfinite model prediction')
                                per=per_scene_srcc(pred,y[te],Ste)
                                score=float(np.mean(per))
                                check(abs(score-h.within_srcc(pred,y[te],Ste))<1e-14,'Metric differs from source S14 helper')
                                pred_table[name]=pred;scene_table[name]=per
                            except Exception as exc:
                                status='FAILED';error=str(exc);score=np.nan
                                pred_table[name]=np.nan;scene_table[name]=np.nan
                                state=dict(error=error,traceback=traceback.format_exc())
                                log.error('%s repeat%d fold%d k%d %s FAILED: %s',ds,rep,fold,k,name,error)
                            elapsed=time.perf_counter()-case_start
                            meta=dict(dataset=ds,model=name,k=k,repeat=rep,fold=fold,status=status,SRCC=score,train_images=len(tr),test_images=len(te),train_scenes=Str.shape[0],test_scenes=Ste.shape[0],fit_human_labels=name!='LOL prior',elapsed_seconds=elapsed,error=error)
                            results.append(meta);attempts.append(meta.copy())
                            if name!='LOL prior':model_file.write(json.dumps(dict(dataset=ds,model=name,k=k,repeat=rep,fold=fold,status=status,**state),allow_nan=False)+'\n')
                        ds_predictions.append(pred_table.assign(dataset=ds,repeat=rep,fold=fold,k=k))
                        ds_scenes.append(scene_table)
                csv('results.csv',pd.DataFrame(results));csv('run_attempts.csv',pd.DataFrame(attempts))
                save('run_status.json',dict(status='RUNNING',dataset=ds,last_repeat=rep,completed=len(results),expected_score_cases=4500,elapsed_seconds=time.perf_counter()-start))
                log.info('%s repeat%d/20 complete; score cases%d/4500',ds,rep+1,len(results))
        predictions=pd.concat(ds_predictions,ignore_index=True);per_scene=pd.concat(ds_scenes,ignore_index=True)
        csv(f'predictions_{ds}.csv',predictions);csv(f'per_scene_{ds}.csv',per_scene)
        check(len(predictions)==config['datasets'][ds]['scenes']*20*5*4,'Prediction count mismatch')
        check(len(per_scene)==config['datasets'][ds]['scenes']*20*5,'Per-scene count mismatch')
    check(snapshot_inputs()==read('input_hashes.json'),'Scientific input changed during fits')
    report=dict(status='MODELS_DONE',finished_utc=now(),score_cases=len(results),human_fits=sum(z['fit_human_labels'] for z in attempts),failed_cases=sum(z['status']!='COMPLETE' for z in results),elapsed_seconds=time.perf_counter()-start)
    save('run_status.json',report)
    log.info('All4500 model score cases processed; failed%d',report['failed_cases'])
if __name__=='__main__':main()
