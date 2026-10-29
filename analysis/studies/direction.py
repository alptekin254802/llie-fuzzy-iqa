"""Controlled degradation direction check; prepare PNGs, run author metrics, summarize."""
from analysis.studies.support_extra import *
import argparse
def prepare():
    root=OUT/'direction_images';root.mkdir(exist_ok=True);rows=[]
    names=sorted(p.name for p in (c.LOL_ROOT/'high').glob('*.png'))[:10]
    if len(names)!=10:raise ValueError('Ten clean LOL reference images are required')
    for i,name in enumerate(names):
        rgb=cv2.imread(str(c.LOL_ROOT/'high'/name))
        rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);noise=np.random.default_rng(SEED+i).standard_normal(rgb.shape)
        cases=[('clean',0,0,rgb)]
        for level,sigma in enumerate([5,10,20,30,50],1):cases.append(('noise',level,sigma,np.clip(np.rint(rgb.astype(float)+sigma*noise),0,255).astype(np.uint8)))
        for level,sigma in enumerate([.5,1,2,3,5],1):
            kernel=2*int(np.ceil(3*sigma))+1
            cases.append(('blur',level,sigma,cv2.GaussianBlur(rgb,(kernel,kernel),sigma,borderType=cv2.BORDER_REFLECT_101)))
        for level,quality in enumerate([90,70,50,30,10],1):
            ok,buffer=cv2.imencode('.jpg',rgb,[cv2.IMWRITE_JPEG_QUALITY,quality]);assert ok
            cases.append(('jpeg',level,quality,cv2.imdecode(buffer,cv2.IMREAD_COLOR)))
        for kind,level,parameter,im in cases:
            key=f'{Path(name).stem}_{kind}_{level}';path=root/(key+'.png')
            assert cv2.imwrite(str(path),im)
            rows.append({'dataset':'direction','name':key,'scene':name,'distortion':kind,'level':level,'parameter':parameter,'path':str(path)})
    pd.DataFrame(rows).to_csv(OUT/'direction_manifest.csv',index=False)
    # A dedicated work_dir is required for this check so no quality dataset manifest is replaced.
    target=OUT/'metric_manifest.csv'
    if target.exists():raise FileExistsError('Use a fresh work_dir for the direction check')
    pd.DataFrame(rows).to_csv(target,index=False)
def analyze():
    manifest=readcsv(OUT/'direction_manifest.csv');result={}
    for metric in ('bpri','bmpri'):
        d=manifest.merge(readcsv(OUT/f'raw_{metric}.csv'),on=['dataset','name'],validate='one_to_one');r={}
        for kind in ('noise','blur','jpeg'):
            correlations=[];ends=[]
            for scene,g in d.groupby('scene'):
                z=g[(g.distortion==kind)|(g.distortion=='clean')].sort_values('level')
                assert len(z)==6
                correlations.append(rho(z.level,z.raw_score));ends.append(float(z.raw_score.iloc[-1]-z.raw_score.iloc[0]))
            r[kind]={'mean_srcc':float(np.mean(correlations)),'positive_scenes':int(np.sum(np.array(correlations)>0)),'endpoint_increase_scenes':int(np.sum(np.array(ends)>0)),'scene_srcc':correlations}
        passed=all(x['mean_srcc']>=.5 and x['positive_scenes']>=8 and x['endpoint_increase_scenes']>=8 for x in r.values())
        result[metric]={'checks':r,'lower_is_better_confirmed':passed,'orientation':-1 if passed else None}
    save(OUT/'metric_directions.json',result)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','analyze']);a=parser.parse_args()
    prepare() if a.stage=='prepare' else analyze()
