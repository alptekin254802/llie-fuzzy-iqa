"""Scene-weighted exploratory descriptions. No hypothesis test or p-value."""
import numpy as np
import pandas as pd
METRICS=['lpips','psnr','ssim']
SELECTORS=['fuzzy','fixed','nr_mono','entropy','brightness','default','oracle','clipiqa+','niqe']
SEED=20261018
REPLICATES=2000

def scene_means(selected):
    if selected.empty or selected.dry_run.any():raise ValueError('Dry-run or empty input cannot enter exploratory descriptions')
    for scene,g in selected.groupby('scene'):
        if g.room_label.nunique()!=1 or g.direction_label.nunique()!=1:raise ValueError('Inconsistent scene labels')
        if set(g.selector)!=set(SELECTORS):raise ValueError('Missing selector')
        for method,x in g.groupby('selector'):
            if len(x)!=20 or set(x.frame)!=set(range(20)):raise ValueError('Incomplete scene; individual frames cannot be omitted')
    return selected.groupby(['scene','room_label','direction_label','selector'],sort=True)[METRICS].mean().reset_index()

def describe_group(scene):
    ids=sorted(scene.scene.unique());n=len(ids)
    rng=np.random.default_rng(SEED);idx=rng.integers(n,size=(REPLICATES,n)) if n>=2 else None
    result={'n_scenes':n,'unit':'scene, equal weight','interpretation':'descriptive only','scene_mean_metrics_by_selector':{},'bootstrap_replicates':REPLICATES,'seed':SEED,'interval_method':'descriptive percentile 2.5/97.5; linear quantiles; scene resampling','interval_unavailable_reason':None if n>=2 else 'fewer than two scenes; no interval'}
    arrays={}
    for method in SELECTORS:
        values=scene[scene.selector==method].set_index('scene').loc[ids,METRICS].to_numpy()
        arrays[method]=values;assert np.isfinite(values).all(),'Nonfinite metric; report and stop, do not omit frames'
        result['scene_mean_metrics_by_selector'][method]={}
        for j,metric in enumerate(METRICS):
            vals=values[:,j];interval=np.quantile(vals[idx].mean(1),[.025,.975]).tolist() if idx is not None else None
            result['scene_mean_metrics_by_selector'][method][metric]={'mean':float(vals.mean()),'descriptive_bootstrap_interval95':interval}
    diff=arrays['fuzzy'][:,0]-arrays['fixed'][:,0]
    result['scene_mean_lpips_difference_fuzzy_minus_fixed']={'mean':float(diff.mean()),'descriptive_bootstrap_interval95':np.quantile(diff[idx].mean(1),[.025,.975]).tolist() if idx is not None else None}
    diff_default=arrays['fuzzy'][:,0]-arrays['default'][:,0]
    result['scene_mean_lpips_difference_fuzzy_minus_default']={'mean':float(diff_default.mean()),'descriptive_bootstrap_interval95':np.quantile(diff_default[idx].mean(1),[.025,.975]).tolist() if idx is not None else None}
    return result

def describe(selected,destination):
    scene=scene_means(selected);n=scene.scene.nunique()
    if n<20:raise ValueError('At least 20 accepted non-dry-run scenes required; no result created')
    labels=scene[['scene','room_label','direction_label']].drop_duplicates()
    if labels.duplicated(['room_label','direction_label']).any():raise ValueError('Duplicate accepted room/direction label')
    scene.to_csv(destination/'scene_means.csv',index=False)
    rooms=sorted(scene.room_label.unique())
    result={'analysis_framework':'exploratory','hypothesis_test_performed':False,'p_values_computed':False,'preregistered_primary_hypothesis':False,'reporting_decision':'Reporting is descriptive; no confirmatory claim.','n_scenes':int(n),'target_scenes':30,'minimum_scenes':20,'rooms':rooms,'overall':describe_group(scene),'by_room':{room:describe_group(scene[scene.room_label==room]) for room in rooms} if len(rooms)>1 else {},'room_interpretation':'Viewing directions within a room may be dependent; descriptive scene bootstrap is not evidence from independent rooms.'}
    scene.groupby(['room_label','selector'],sort=True)[METRICS].mean().reset_index().to_csv(destination/'room_means.csv',index=False)
    return result
