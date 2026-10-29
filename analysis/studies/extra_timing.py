"""Isolated latency for BPRI, BMPRI and StairIQA. Excludes image read and model load."""
from analysis.studies.support_extra import *
import re, shutil, importlib.util

def main():
    import torch
    from torchvision import transforms
    from PIL import Image
    result = {'started': now(), 'protocol': '50 images, 3 warmups, 1 CPU thread; sequential isolated measurements; set and document the power profile externally', 'metrics': {}}
    import psutil
    result['ambient_cpu_percent'] = [psutil.cpu_percent(interval=1) for _ in range(5)]
    manifest = readcsv(OUT / 'metric_manifest.csv').iloc[:50]
    mp = OUT / 'isolated_timing_manifest.csv'
    manifest.to_csv(mp, index=False)
    for metric in ['bpri', 'bmpri']:
        work = OUT / f'matlab_timing_{metric}'
        if not work.exists():
            shutil.copytree(c.THIRD_PARTY[metric], work)
        out = OUT / f'isolated_{metric}.json'
        script = OUT / f'isolated_{metric}.m'
        code = f"maxNumCompThreads(1);warning('off','all');addpath(genpath('{work.as_posix()}'));cd('{work.as_posix()}');T=readtable('{mp.as_posix()}','TextType','string');im=imread(T.path(1));for j=1:3,z={metric.upper()}(im);end\nts=[];scores=[];for i=1:height(T),im=imread(T.path(i));tic;scores(end+1)={metric.upper()}(im);ts(end+1)=toc*1000;if i>=5&&mean(ts)>5000,break;end;end\nr=struct('version',version,'threads',maxNumCompThreads,'milliseconds',ts,'scores',scores,'finished',char(datetime('now')));fid=fopen('{out.as_posix()}','w');fprintf(fid,'%s',jsonencode(r));fclose(fid);\n"
        script.write_text(code, encoding='utf-8')
        with (OUT / f'isolated_{metric}.log').open('w') as log:
            subprocess.run([c.MATLAB, '-batch', f"run('{script.as_posix()}')"], stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        r = json.loads(out.read_text())
        result['metrics'][metric] = r | {'timing': stats_ms(r['milliseconds'])}
        save(OUT / 'extra_metric_timing_checkpoint.json', result)
        print(metric, result['metrics'][metric]['timing'], flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = c.THIRD_PARTY['stairiqa']
    spec = importlib.util.spec_from_file_location('stair_timing_author', source / 'models/ResNet_staircase.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.resnet50(pretrained=False)
    state = torch.load(c.STAIR_WEIGHTS, map_location='cpu', weights_only=True)
    net.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    net.eval()
    trans = transforms.Compose([transforms.Resize(384), transforms.FiveCrop(320), lambda crops: torch.stack([transforms.ToTensor()(x) for x in crops]), lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x) for x in crops])])
    images = [Image.open(p).convert('RGB') for p in manifest.path]
    with torch.inference_mode():
        for device in ['cpu', 'cuda']:
            net.to(device)
            ts = []
            scores = []
            for j in range(3):
                net(trans(images[j]).to(device))
            for im in images:
                if device == 'cuda':
                    torch.cuda.synchronize()
                t = time.perf_counter()
                scores.append(float(net(trans(im).to(device))[3].mean().item()))
                if device == 'cuda':
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
                if len(ts) >= 5 and np.mean(ts) > 5000:
                    break
            result['metrics']['stairiqa_' + device] = {'timing': stats_ms(ts), 'milliseconds': ts, 'scores': scores, 'status': 'complete' if len(ts) == 50 else 'stopped_above5s'}
            save(OUT / 'extra_metric_timing_checkpoint.json', result)
            print(device, stats_ms(ts), flush=True)
    result['finished'] = now()
    save(OUT / 'extra_metric_timing.json', result)
if __name__ == '__main__':
    main()
