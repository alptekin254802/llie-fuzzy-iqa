"""Run BPRI, BMPRI and StairIQA on the fixed comparison images."""
from analysis.studies.support_extra import *
from analysis.studies.images import human_meta, enhanced, rgb8
import shutil, argparse

def prepare():
    if (OUT / 'metric_manifest.csv').exists() and (OUT / 'metric_input.json').exists():
        existing = readcsv(OUT / 'metric_manifest.csv')
        assert len(existing) == 7865 and all((Path(p).exists() for p in existing.path))
        print('Prepared manifest already complete; preserved', flush=True)
        return
    begin2('extra_interfaces')
    folder = OUT / 'metric_images'
    folder.mkdir(exist_ok=True)
    rows = []
    qmax = 0.0
    for ds in ['lol', 'squarelol', 'lieq', 'rlie']:
        sub = folder / ds
        sub.mkdir(exist_ok=True)
        if ds == 'lol':
            lol = c.lol_table()
            for i, r in lol.iterrows():
                name = r.filename + '__' + r.method
                path = sub / (name + '.png')
                if not path.exists():
                    im = enhanced(c.load_rgb01(c.PREPROCESSED / 'low' / r.filename, 'native'), r.method)
                    u = rgb8(im)
                    qmax = max(qmax, float(np.max(np.abs(im - u.astype(float) / 255))))
                    cv2.imwrite(str(path), cv2.cvtColor(u, cv2.COLOR_RGB2BGR))
                rows.append({'dataset': ds, 'name': name, 'filename': r.filename, 'scene': r.filename, 'method': r.method, 'target': -r.lpips, 'path': str(path)})
        else:
            for _, r in human_meta(ds).iterrows():
                path = sub / r['name']
                if not path.exists():
                    u = cv2.imread(r.path)
                    u = cv2.resize(u, (256, 256), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(path), u)
                rows.append({'dataset': ds, 'name': r['name'], 'filename': r['name'], 'scene': r.scene, 'method': r.method, 'target': r.mos, 'path': str(path)})
    x = pd.DataFrame(rows)
    x.to_csv(OUT / 'metric_manifest.csv', index=False)
    save(OUT / 'metric_input.json', {'rows': len(x), 'dataset_counts': x.dataset.value_counts().to_dict(), 'lol_float_to_uint8_max_abs': qmax, 'manifest_sha256': sha(OUT / 'metric_manifest.csv')})
    print('Prepared', len(x), flush=True)

def matlab(metric):
    source = c.THIRD_PARTY[metric]
    work = OUT / f'matlab_work_{metric}'
    if not work.exists():
        shutil.copytree(source, work)
    for p in source.glob('*.m'):
        assert sha(p) == sha(work / p.name)
    fn = metric.upper()
    manifest = OUT / 'metric_manifest.csv'
    out = OUT / f'raw_{metric}.csv'
    status = OUT / f'matlab_{metric}_status.json'
    code = f"maxNumCompThreads(1);\nwarning('off','all');\naddpath(genpath('{work.as_posix()}')); cd('{work.as_posix()}');\nT=readtable('{manifest.as_posix()}','TextType','string');\noutfile='{out.as_posix()}'; start=1;\nif isfile(outfile), D=readtable(outfile); start=height(D)+1; end\ninfo=struct('metric','{metric}','matlab_version',version,'threads',maxNumCompThreads,'started',char(datetime('now')),'status','running');\ntry\n im=imread(T.path(1)); for j=1:3, z={fn}(im); end\n fid=fopen(outfile,'a'); if start==1, fprintf(fid,'dataset,name,raw_score,sec\\n'); end\n dtall=[];\n for i=start:height(T)\n  im=imread(T.path(i)); tic; score={fn}(im); dt=toc;\n  assert(isscalar(score)&&isfinite(score),'Invalid score');\n  fprintf(fid,'%s,%s,%.17g,%.17g\\n',T.dataset(i),T.name(i),score,dt); dtall(end+1)=dt;\n  if mod(i,25)==0, fclose(fid);fid=fopen(outfile,'a');fprintf('%s %d/%d %.3fs\\n','{metric}',i,height(T),mean(dtall(max(1,end-24):end)));end\n  if length(dtall)>=10 && mean(dtall(max(1,end-9):end))>5,info.status='stopped_above5s';break;end\n end\n fclose(fid);info.completed=i;info.timing_seconds=dtall;\n if i==height(T),info.status='complete';end\ncatch ME\n info.status='error';info.error=getReport(ME,'extended','hyperlinks','off');disp(info.error);\nend\ninfo.finished=char(datetime('now'));fid=fopen('{status.as_posix()}','w');fprintf(fid,'%s',jsonencode(info));fclose(fid);\n"
    script = OUT / f'matlab_{metric}.m'
    script.write_text(code, encoding='utf-8')
    with (OUT / f'matlab_{metric}.log').open('a', encoding='utf-8') as log:
        proc = subprocess.run([c.MATLAB, '-batch', f"run('{script.as_posix()}')"], stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    if proc.returncode != 0:
        raise RuntimeError('MATLAB process failed; inspect the saved log')
    state=json.loads(status.read_text())
    if state['status'] != 'complete':
        raise RuntimeError('MATLAB metric incomplete: '+state['status'])
    print(metric, 'MATLAB complete', flush=True)

def stair():
    import torch
    from torchvision import transforms
    from PIL import Image
    import importlib.util
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = c.THIRD_PARTY['stairiqa']
    spec = importlib.util.spec_from_file_location('stair_author', source / 'models/ResNet_staircase.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = torch.nn.DataParallel(mod.resnet50(pretrained=False))
    weights = c.STAIR_WEIGHTS
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    transform = transforms.Compose([transforms.Resize(384), transforms.FiveCrop(320), lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]), lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])])
    manifest = readcsv(OUT / 'metric_manifest.csv', dtype={'scene': str})
    output = OUT / 'raw_stairiqa.csv'
    rows = readcsv(output).to_dict('records') if output.exists() else []
    done = {(r['dataset'], r['name']) for r in rows}

    def infer(path, device):
        x = transform(Image.open(path).convert('RGB')).to(device)
        net = model.module if device == 'cpu' else model
        return float(net(x)[3].mean().item())
    timing = {'training': 'author IMDT checkpoint with KonIQ10k regression head', 'weights_sha256': sha(weights), 'source_commit': '0ed6d099fd057bb85c3675788c539cd70c346db9', 'started': now(), 'torch': torch.__version__}
    with torch.inference_mode():
        model.to('cpu')
        cpu = []
        for p in manifest.path[:3]:
            infer(p, 'cpu')
        for p in manifest.path[:10]:
            t = time.perf_counter()
            infer(p, 'cpu')
            cpu.append((time.perf_counter() - t) * 1000)
            if len(cpu) >= 3 and np.mean(cpu) > 5000:
                break
        timing['cpu_ms'] = stats_ms(cpu)
        timing['cpu_status'] = 'stopped_above5s' if np.mean(cpu) > 5000 else 'complete10'
        save(OUT / 'stair_timing.json', timing)
        model.to('cuda')
        for p in manifest.path[:3]:
            infer(p, 'cuda')
        torch.cuda.synchronize()
        recent = []
        for i, r in manifest.iterrows():
            if (r.dataset, r['name']) in done:
                continue
            torch.cuda.synchronize()
            t = time.perf_counter()
            score = infer(r.path, 'cuda')
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            recent.append(dt)
            rows.append({'dataset': r.dataset, 'name': r['name'], 'raw_score': score, 'sec': dt})
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(output, index=False)
                print('StairIQA', len(rows), '/', len(manifest), 'sec', np.mean(recent[-25:]), flush=True)
            if len(recent) >= 10 and np.mean(recent[-10:]) > 5:
                timing['gpu_status'] = 'stopped_above5s'
                break
        pd.DataFrame(rows).to_csv(output, index=False)
        timing['gpu_ms'] = stats_ms(np.array([r['sec'] for r in rows]) * 1000)
        timing['gpu_status'] = 'complete' if len(rows) == len(manifest) else timing.get('gpu_status', 'incomplete')
        timing['finished'] = now()
        save(OUT / 'stair_timing.json', timing)

def analyze():
    meta = begin2('extra_metrics')
    manifest = readcsv(OUT / 'metric_manifest.csv', dtype={'scene': str})
    res = {'metrics': {}, 'datasets': {}, 'orientation': {'bpri': -1, 'bmpri': -1, 'stairiqa': 1}, 'matlab_training_provenance': 'Author releases contain model/range; no explicit checkpoint dataset documentation. BMPRI paper cross-dataset model trained LIVE, not independently confirmed for binary.'}
    for m in ['bpri', 'bmpri', 'stairiqa']:
        path = OUT / f'raw_{m}_parallel.csv'
        if not path.exists():
            path = OUT / f'raw_{m}.csv'
        if not path.exists():
            res['metrics'][m] = {'status': 'unavailable'}
            continue
        x = readcsv(path)
        res['metrics'][m] = {'rows': int(x.raw_score.notna().sum()), 'status': 'complete' if len(x) == len(manifest) and x.raw_score.notna().all() else 'partial', 'timing': stats_ms(x.sec.dropna() * 1000)}
        manifest = manifest.merge(x[['dataset', 'name', 'raw_score']].rename(columns={'raw_score': m}), on=['dataset', 'name'], how='left')
    sg, mfs, th = c.lol_frozen_model()
    for ds, x in manifest.groupby('dataset', sort=False):
        if ds == 'lol':
            feat = c.lol_table()
            feat['name'] = feat.filename + '__' + feat.method
        else:
            feat = readcsv(OUT / f'feat_{ds}_area256.csv', dtype={'scene': str})
        x = x.merge(feat[['name'] + c.NR], on='name')
        x['fuzzy'] = c.fuzzy_predict(x, sg, mfs, th)
        if ds == 'lol':
            oo = readcsv(OUT / 'validation_inputs.csv')
            oo['name'] = oo.filename + '__' + oo.method
            x['fuzzy'] = x['name'].map(oo.set_index('name').oof_nr_free)
        x.to_csv(OUT / f'extra_metrics_{ds}.csv', index=False)
        rr = {}
        for m in ['fuzzy', 'bpri', 'bmpri', 'stairiqa']:
            if m not in x or x[m].isna().any():
                continue
            p = x[m].to_numpy() * res['orientation'].get(m, 1)
            yy = x.target.to_numpy()
            gg = x.scene.to_numpy()
            r = {'pooled_srcc': rho(p, yy), 'within': within2(p, yy, gg)}
            if m != 'fuzzy':
                r['within_srcc_minus_fuzzy'] = paired_within(p, x.fuzzy, yy, gg)
                diffs = []
                npairs = []
                tops = []
                for scene, z in x.assign(pred=p).groupby('scene', sort=True):
                    aa = within2(z.pred, z.target, np.repeat(scene, len(z)))
                    bb = within2(z.fuzzy, z.target, np.repeat(scene, len(z)))
                    diffs.append((aa['pair_acc'] - bb['pair_acc']) * aa['pairs'])
                    npairs.append(aa['pairs'])
                    tops.append(aa['top1'] - bb['top1'])
                diffs = np.array(diffs)
                npairs = np.array(npairs)
                rng = np.random.default_rng(SEED)
                draw = []
                for _ in range(B):
                    idx = rng.integers(len(diffs), size=len(diffs))
                    draw.append(diffs[idx].sum() / npairs[idx].sum())
                r['pair_acc_minus_fuzzy'] = interval(draw, float(diffs.sum() / npairs.sum()))
                r['top1_minus_fuzzy'] = diff_ci(tops)
                if ds == 'lol':
                    r['pooled_minus_fuzzy'] = paired_bootstrap(p, x.fuzzy, yy, gg)
            rr[m] = r
        res['datasets'][ds] = rr
    finish2(meta, res, [OUT / f'extra_metrics_{ds}.csv' for ds in ['lol', 'squarelol', 'lieq', 'rlie']])
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'bpri', 'bmpri', 'stairiqa', 'analyze'])
    a = ap.parse_args()
    if a.stage == 'prepare':
        prepare()
    elif a.stage in ['bpri', 'bmpri']:
        matlab(a.stage)
    elif a.stage == 'stairiqa':
        stair()
    else:
        analyze()
