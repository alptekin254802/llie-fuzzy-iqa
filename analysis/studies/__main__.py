"""Run a reported study in a separate local workspace."""
import argparse, os, sys, runpy, json
from pathlib import Path
COMMANDS={
 'qalign':'qalign','richiqa':'richiqa',
 'prepare':'prepare','features':'features','brisque-features':'brisque_features','lol-brisque':'lol_brisque',
 'human-transfer':'human_transfer','human-metrics':'human_metrics','human-metric-summary':'human_metric_summary','human-metric-uncertainty':'human_metric_uncertainty',
 'human-recalibration':'human_recalibration','human-method-disjoint':'human_method_disjoint',
 'proxy-brisque':'proxy_brisque','transfer-uncertainty':'transfer_uncertainty',
 'capacity-ablation':'capacity_ablation','scene-overlap':'scene_overlap','overlap-transfer':'overlap_transfer',
 'method-mean':'method_mean','regional':'regional','de-sensitivity':'de_sensitivity','retinex':'retinex',
 'native':'native_resolution','closed-loop':'closed_loop','extra-metrics':'extra_metrics','direction':'direction','extra-timing':'extra_timing',
 'monotonicity':'monotonicity','canonical-human':'canonical_human','rule-support':'rule_support'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('study',choices=COMMANDS)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    a=parser.parse_args()
    os.environ['LLIE_STUDY_CONFIG']=str(a.config.resolve())
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    from analysis.studies.config import OUT,REPO
    if OUT.resolve()==REPO.resolve() or any(OUT.resolve().is_relative_to(REPO/sub) for sub in ('results','models')):
        raise ValueError('Use a separate work_dir, not the shipped results or models directories')
    OUT.mkdir(parents=True,exist_ok=True)
    import hashlib
    config_bytes=a.config.read_bytes();stamp=OUT/'workspace_config_sha256.txt'
    signature=hashlib.sha256(config_bytes).hexdigest()
    if stamp.exists() and stamp.read_text()!=signature:
        raise ValueError('Configuration changed; use a fresh work_dir to avoid mixing cached datasets')
    stamp.write_text(signature)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    # Reject partial resumable caches before model fitting or statistical evaluation.
    import csv
    sizes={'squarelol':2900,'lieq':1000,'rlie':1540}
    all_sets={'capacity-ablation','proxy-brisque','transfer-uncertainty','regional','native','canonical-human','human-metric-summary','human-metric-uncertainty','extra-metrics'}
    single_sets={'human-transfer','human-recalibration','human-method-disjoint'}
    required=list(sizes) if a.study in all_sets and not (a.study=='extra-metrics' and a.arguments and a.arguments[0] in ('bpri','bmpri','stairiqa')) else ([a.arguments[0]] if a.study in single_sets and a.arguments else [])
    if a.study in {'de-sensitivity','rule-support','overlap-transfer'}:required=['squarelol']
    for ds in required:
        modes=('area256','native') if a.study=='native' else ('area256',)
        for mode in modes:
            path=OUT/f'feat_{ds}_{mode}.csv'
            if not path.exists():raise FileNotFoundError(f'Run features {ds} {mode} first')
            with path.open(newline='') as handle:rows=list(csv.DictReader(handle))
            if len(rows)!=sizes[ds] or len({r['name'] for r in rows})!=sizes[ds]:
                raise ValueError(f'{path.name} is incomplete or has duplicate image names')
    sys.argv=[a.study]+a.arguments
    runpy.run_module('analysis.studies.'+COMMANDS[a.study],run_name='__main__')
if __name__=='__main__':main()
