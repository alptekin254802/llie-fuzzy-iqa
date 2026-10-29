"""Reproduce VQA2/VITAL scores from local images and pinned author releases."""
from pathlib import Path
import argparse, csv, hashlib, json, os
ROOT=Path(__file__).resolve().parents[2]
RECORDS=ROOT/'results/lmm_extra'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_files(folder,manifest):
    for name,expected in json.loads(manifest.read_text(encoding='utf-8')).items():
        p=folder/name
        if not p.is_file() or digest(p)!=expected:
            raise ValueError('Pinned release file mismatch: '+name)

def resolve_config(path,model):
    raw=json.loads(path.read_text(encoding='utf-8-sig'))
    def resolve(value):return (path.parent/value).resolve()
    return {k:resolve(v) for k,v in raw[model].items()},{k:resolve(v) for k,v in raw['datasets'].items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['vqa2','vital'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=['nf4_fp32','bf16_cpu','fp16'],default='nf4_fp32')
    p.add_argument('--limit',type=int,help='Optional smoke-test prefix; omit for all images')
    p.add_argument('--pilot',action='store_true',help='Use the recorded 20-image pilot order')
    p.add_argument('--pilot-decision',type=Path,help='Accepted local precision/direction check required for a full run')
    p.add_argument('--verify-assets-only',action='store_true')
    a=p.parse_args();a.config=a.config.resolve();a.output=a.output.resolve()
    # Scores and offload caches cannot overwrite distributed results or source files.
    if a.output==ROOT or (ROOT in a.output.parents and ROOT/'study_work' not in a.output.parents):
        p.error('Use study_work/<run> or an external output directory')
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    paths,roots=resolve_config(a.config,a.model)
    check_files(paths['source'],RECORDS/a.model/'source_hashes.json')
    check_files(paths['checkpoint'],RECORDS/a.model/'checkpoint_hashes.json')
    if a.model=='vqa2':
        expected=json.loads((RECORDS/'asset_hashes.json').read_text())['slowfast.pth']
        if digest(paths['slowfast'])!=expected:raise ValueError('SlowFast checkpoint mismatch')
    if a.verify_assets_only:print('Pinned source and checkpoint files: PASS');return
    if a.precision=='bf16_cpu' and not a.pilot:p.error('Use --pilot for the CPU reference')
    if not a.pilot:
        if a.pilot_decision is None:p.error('Full inference requires --pilot-decision from analysis.lmm_extra.verify')
        decision=json.loads(a.pilot_decision.read_text(encoding='utf-8'))
        if not decision.get('accepted') or decision.get('model')!=a.model or decision.get('precision')!=a.precision:
            raise ValueError('No accepted matching pilot decision')
    threads=8 if a.precision=='bf16_cpu' else 1
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]=str(threads)
    a.output.mkdir(parents=True,exist_ok=True)
    for key,sub in [('HF_HOME','hf_cache'),('TORCH_HOME','torch_cache'),('CUDA_CACHE_PATH','cuda_cache'),('TMP','tmp'),('TEMP','tmp')]:
        os.environ[key]=str(a.output/sub);Path(os.environ[key]).mkdir(exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LMM_DEVICE_MAP='auto')
    import numpy as np,cv2,torch,random,time,importlib.metadata as metadata
    from .adapters import load
    from .scoring import quality
    random.seed(20261018);np.random.seed(20261018);torch.manual_seed(20261018)
    cv2.setNumThreads(1);torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    if a.precision!='bf16_cpu' and not torch.cuda.is_available():raise RuntimeError('CUDA required for this precision')
    manifest=RECORDS/a.model/'pilot_reference.csv' if a.pilot else RECORDS/'image_manifest.csv'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8',newline='')))
    if a.limit:rows=rows[:a.limit]
    identity=[(r['dataset'],r['name'],r['rgb256_sha256']) for r in rows]
    config={'model':a.model,'precision':a.precision,'pilot':a.pilot,'identity':identity,'threads':threads}
    config_file=a.output/'run_config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:raise ValueError('Resume configuration differs')
    config_file.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    dest=a.output/'scores.csv';previous=[]
    if dest.exists():
        if not dest.read_bytes().endswith(b'\n'):raise ValueError('Incomplete CSV tail; preserve and repair it before resuming')
        previous=list(csv.DictReader(dest.open(encoding='utf-8',newline='')))
        if [(r['dataset'],r['name'],r['rgb256_sha256']) for r in previous]!=identity[:len(previous)]:raise ValueError('Resume is not an exact input prefix')
        for r in previous:
            if r['precision']!=a.precision or abs(quality(json.loads(r['level_logits_json']))-float(r['score']))>1e-12:raise ValueError('Invalid existing score')
    if len(previous)==len(rows):print('All requested scores already present and verified');return
    model,scorer,info=load(a.model,a.precision,paths,a.output)
    info['versions']={k:metadata.version(k) for k in ['torch','torchvision','transformers','accelerate','bitsandbytes','numpy','opencv-python','timm']}
    info['threads']=threads
    (a.output/'environment.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    def evaluate(row):
        wall=time.perf_counter();image_path=roots[row['dataset']]/row['name']
        bgr=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if bgr is None:raise FileNotFoundError(image_path)
        rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(256,256),interpolation=cv2.INTER_AREA)
        h=hashlib.sha256(rgb.tobytes()).hexdigest()
        if h!=row['rgb256_sha256']:raise ValueError('Input RGB hash mismatch: '+row['name'])
        x=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
        prep=(time.perf_counter()-wall)*1000;start=time.perf_counter()
        if a.precision!='bf16_cpu':x=x.to('cuda');torch.cuda.synchronize()
        h2d=(time.perf_counter()-start)*1000;start=time.perf_counter()
        score,logits,diagnostic=scorer(rgb)
        if a.precision!='bf16_cpu':torch.cuda.synchronize()
        inference=(time.perf_counter()-start)*1000
        if not np.isfinite(score) or not np.isfinite(logits).all():raise FloatingPointError('Nonfinite score; inference stopped')
        return {'dataset':row['dataset'],'name':row['name'],'scene':row['scene'],'score':score,'prepare_ms':prep,'h2d_ms':h2d,'inference_ms':inference,'wall_ms':(time.perf_counter()-wall)*1000,'rgb256_sha256':h,'precision':a.precision,'level_logits_json':json.dumps(logits),'source_key_defect_score':diagnostic}
    with torch.inference_mode():
        if not a.pilot:
            allrows=list(csv.DictReader((RECORDS/'image_manifest.csv').open(encoding='utf-8',newline='')))
            warm=[evaluate(row) for dataset in ['squarelol','lieq','rlie'] for row in [r for r in allrows if r['dataset']==dataset][:3]]
            (a.output/'warmup.json').write_text(json.dumps({'excluded':True,'calls':warm},indent=2)+'\n',encoding='utf-8')
        fields=['dataset','name','scene','score','prepare_ms','h2d_ms','inference_ms','wall_ms','rgb256_sha256','precision','level_logits_json','source_key_defect_score']
        with dest.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            if f.tell()==0:writer.writeheader()
            for row in rows[len(previous):]:
                record=evaluate(row);writer.writerow(record);f.flush();os.fsync(f.fileno())
                print(row['dataset'],row['name'],record['score'],flush=True)

if __name__=='__main__':main()
