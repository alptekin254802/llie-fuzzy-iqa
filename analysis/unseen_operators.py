"Portable reproduction of the exploratory unseen-enhancement check."
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.stats import rankdata,spearmanr
from threadpoolctl import threadpool_limits
from analysis import tier3_broaden as tier
from evaluation.frozen_scorer import FrozenScorer
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RECORDS=ROOT/'results/studies/unseen_operators'
METHODS={'SSR-15':'ssr_15','SSR-80':'ssr_80','SSR-250':'ssr_250','MSR':'msr','MSRCR':'msrcr','AGCWD':'agcwd','Zero-DCE++':'zero_dce_pp','RUAS-MIT':'ruas_mit'}
FEATURES=['entropy','contrast','sharpness']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\r\n')
def score(model,frame):
    arr=[-frame[k].to_numpy(float) if model['signs'][k] else frame[k].to_numpy(float) for k in FEATURES]
    fire,total=tier.firing(arr,model['mfs'])
    return (fire@np.array(model['theta']))/total

def boot(scores,target,indices):
    a=rankdata(scores[indices],axis=1);b=rankdata(target[indices],axis=1)
    a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    values=np.divide((a*b).sum(axis=1),den,out=np.zeros(len(indices)),where=den!=0)
    return np.quantile(values,[.025,.975]).tolist()

def verify(args):
    records=Path(args.records);reference=pd.read_csv(records/'results.csv',float_precision='round_trip')
    four=load(records/'four_classical_model_posthoc.json');five=load(records/'classical_pool_model.json')
    frozen=FrozenScorer('nr_free');indices=np.random.default_rng(42).integers(0,485,size=(2000,485));checked=[]
    for name,slug in METHODS.items():
        f=pd.read_csv(records/f'predictions_{slug}.csv',float_precision='round_trip')
        assert len(f)==485 and f.filename.nunique()==485 and f.filename.tolist()==sorted(f.filename)
        assert f.method.eq(name).all()
        r=reference[(reference.method==name)&(reference.status=='complete')].iloc[0];target=-f.lpips.to_numpy(float)
        assert np.isfinite(f[FEATURES+['lpips','brightness','noise']]).all().all()
        np.testing.assert_allclose(score(four,f),f.four_classical_score_posthoc,rtol=0,atol=1e-9)
        np.testing.assert_allclose(score(five,f),f.classical_pool_score,rtol=0,atol=1e-9)
        np.testing.assert_allclose(frozen.predict(f[FEATURES]),f.frozen_score,rtol=0,atol=1e-9)
        values={}
        for column,prefix in [('four_classical_score_posthoc','four_classical'),('classical_pool_score','classical_pool'),('frozen_score','frozen')]:
            a=f[column].to_numpy(float);rho=tier.rho(a,target);ci=boot(a,target,indices)
            assert abs(rho-r[prefix+'_srcc'])<1e-12
            np.testing.assert_allclose(ci,[r[prefix+'_ci_low'],r[prefix+'_ci_high']],rtol=0,atol=1e-12)
            values[prefix]={'srcc':rho,'ci95':ci}
        for column,key,sign in [('entropy','entropy_srcc',1),('contrast','contrast_srcc',1),('sharpness','sharpness_srcc',-1),('brightness','brightness_srcc_posthoc',1),('noise','neg_noise_srcc_posthoc',-1)]:
            value=tier.rho(sign*f[column].to_numpy(float),target);assert abs(value-r[key])<1e-12;values[key]=value
        checked.append({'method':name,'n':485,'values':values})
    nf=pd.read_csv(ROOT/'results/nr_features.csv')[['filename','method']+FEATURES]
    pc=pd.read_csv(ROOT/'results/perceptual_metrics.csv')[['filename','method','lpips']]
    train=nf.merge(pc,on=['filename','method']).dropna();ssr=train[train.method=='retinex']
    gate=tier.rho(score(four,ssr),-ssr.lpips.to_numpy());assert abs(gate-(-.22124266699168893))<1e-12
    report={'status':'PASS','ssr50_control':gate,'methods':checked,'bootstrap':{'draws':2000,'seed':42,'unit':'scene','refit':False}}
    if args.output:save(args.output,report)
    print(json.dumps({'status':'PASS','methods':len(checked),'ssr50_control':gate,'intervals_and_all_single_feature_columns_match':True},indent=2))

def calibrate(args):
    nf=pd.read_csv(ROOT/'results/nr_features.csv')[['filename','method']+FEATURES]
    pc=pd.read_csv(ROOT/'results/perceptual_metrics.csv')[['filename','method','lpips']]
    train=nf.merge(pc,on=['filename','method']).dropna()
    if args.pool==4:train=train[train.method!='retinex']
    assert len(train)==args.pool*485
    target=-train.lpips.to_numpy();signs={k:bool(spearmanr(train[k],target).statistic<0) for k in FEATURES}
    arr=[-train[k].to_numpy(float) if signs[k] else train[k].to_numpy(float) for k in FEATURES]
    mfs=[tier.mfset(a) for a in arr];fire,total=tier.firing(arr,mfs)
    theta=tier.calibrate(fire,total,target,np.arange(len(train)))
    save(args.output,{'signs':signs,'mfs':mfs,'theta':theta.tolist(),'t_norm':'minimum','training_rows':len(train),'posthoc':args.pool==4})

# MSR averages log-surround responses before normalisation. AGCWD operates on HSV V.
def msr(low):
            out = np.zeros_like(low)
            for sigma in (15, 80, 250):
                tmp = np.zeros_like(low)
                for channel in range(3):
                    v = low[:, :, channel]
                    tmp[:, :, channel] = np.log(v + 1e-8) - np.log(cv2.GaussianBlur(v, (0, 0), sigma) + 1e-8)
                out += tmp / 3
            out -= out.min()
            out /= out.max() + 1e-8
            return out

def agcwd(low):
            rgb8 = np.clip(low * 255, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
            pdf = np.bincount(v.ravel(), minlength=256).astype(np.float64) / v.size
            pmax, pmin = pdf.max(), pdf.min()
            if pmax == pmin:
                return low.copy()
            wp = pmax * ((pdf - pmin) / (pmax - pmin)) ** .5
            cdf = np.cumsum(wp) / wp.sum()
            lut = (255 * (np.arange(256) / 255) ** (1 - cdf)).astype(np.uint8)
            lut[0] = 0
            hsv[:, :, 2] = lut[v]
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def external_files(root,records,folder):
    assets=[x for x in load(Path(records)/'external_sources.json') if x['local'].startswith(folder+'/')]
    for x in assets:
        p=Path(root)/x['local']
        if not p.is_file():raise FileNotFoundError(f'{p}: obtain the pinned upstream file documented in README.md')
        if sha(p)!=x['sha256']:raise ValueError(f'Upstream file hash mismatch: {p}')
    return Path(root)/folder

def enhancer(args):
    from preprocessing.enhancement.methods import apply_retinex
    if args.method.startswith('SSR-'):return lambda x:apply_retinex(x,sigma=int(args.method.split('-')[1]))
    if args.method=='MSR':return msr
    if args.method=='AGCWD':return agcwd
    if args.method=='MSRCR':
        p=external_files(args.external_root,args.records,'retinex');m=module('external_retinex',p/'retinex.py');cfg=load(p/'config.json')
        return lambda x:m.MSRCR(np.clip(x*255,0,255).astype(np.uint8),**cfg).astype(np.float32)/255
    import torch
    from evaluation.perceptual_metrics import to_tensor
    if args.method=='Zero-DCE++':
        p=external_files(args.external_root,args.records,'zero_dce_pp');m=module('external_zero',p/'model.py');net=m.enhance_net_nopool(scale_factor=1).cpu()
        net.load_state_dict(torch.load(p/'Epoch99.pth',map_location='cpu',weights_only=True),strict=True);net.eval()
        def run(x):
            with torch.no_grad():y,_=net(to_tensor(x,'cpu'))
            return y[0].permute(1,2,0).cpu().numpy()
    else:
        p=external_files(args.external_root,args.records,'ruas');sys.path.insert(0,str(p))
        from model_enhance.model import Network
        net=Network().cpu();net.load_state_dict(torch.load(p/'weights/mit.pt',map_location='cpu',weights_only=True),strict=True);net.eval()
        def run(x):
            with torch.no_grad():y,_,_=net(to_tensor(x,'cpu'))
            return y[-1][0].permute(1,2,0).cpu().numpy()
    return run

def measure(args):
    import torch,pyiqa
    from evaluation.nr_features import nr_features,load_rgb_float
    from evaluation.perceptual_metrics import to_tensor
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
    root=Path(args.data_root);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    names=sorted(pd.read_csv(ROOT/'results/nr_features.csv').filename.unique());assert len(names)==485
    fn=enhancer(args);metric=pyiqa.create_metric('lpips',device='cpu');assert metric.net.pnet_type=='alex' and metric.net.version=='0.1' and not metric.net.training
    results=[]
    for name in names:
        low=load_rgb_float(root/'low'/name);high=load_rgb_float(root/'high'/name)
        assert low.shape==high.shape==(256,256,3)
        enhanced=np.clip(fn(low),0,1).astype(np.float32);assert enhanced.shape==low.shape and np.isfinite(enhanced).all()
        features=nr_features(enhanced)
        with torch.no_grad():distance=float(metric(to_tensor(enhanced,'cpu'),to_tensor(high,'cpu')).item())
        assert np.isfinite(distance)
        results.append({'filename':name,'method':args.method,**features,'lpips':distance})
    out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(out,index=False)
    print(f'Wrote {len(results)} measurements; no image or weight files written.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('verify');p.add_argument('--records',default=str(DEFAULT_RECORDS));p.add_argument('--output')
    p=sub.add_parser('calibrate');p.add_argument('--pool',type=int,choices=[4,5],required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('measure');p.add_argument('--method',choices=list(METHODS)+['SSR-50'],required=True);p.add_argument('--data-root',required=True);p.add_argument('--external-root',required=True);p.add_argument('--records',default=str(DEFAULT_RECORDS));p.add_argument('--output',required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):globals()[args.command](args)
if __name__=='__main__':main()
