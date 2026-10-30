"""Recheck confirmation eligibility using normal-light reference pixels only."""
import argparse,collections,hashlib,zipfile
import cv2,numpy as np
from .common import DATA,configure,resolve,read,csv,output_dir

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();configure(a.config);out=output_dir(a.output)
    candidates=read(DATA/'confirmation/reference_classification.csv').fillna('')
    prior=read(DATA/'confirmation/prior_references.csv').fillna('');cache={}
    with zipfile.ZipFile(resolve('lolv2_archive')) as archive:
        def image(path,member,expected):
            key=(path,member)
            if key in cache:return cache[key]
            raw=resolve(path).read_bytes() if path else archive.read(member)
            h=hashlib.sha256(raw).hexdigest()
            if h!=expected:raise ValueError('Reference bytes differ: '+str(key))
            bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            rh=hashlib.sha256(('x'.join(map(str,rgb.shape))).encode()+b'\0'+rgb.tobytes()).hexdigest()
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            v=cv2.equalizeHist(cv2.resize(gray,(48,32),interpolation=cv2.INTER_AREA)).astype(np.float64).ravel()
            cache[key]=((v-v.mean())/(v.std()+1e-9),h,rh)
            return cache[key]
        first=[image('',r['reference_member'],r['reference_sha256']) for r in candidates.to_dict('records')]
        earlier=[image(r['path'],r['member'],r['file_sha256']) for r in prior.to_dict('records')]
    X=np.stack([x[0] for x in first]);B=np.stack([x[0] for x in earlier]);cross=X@B.T/X.shape[1];within=X@X.T/X.shape[1]
    parent=list(range(len(first)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i in range(len(first)):
        for j in np.flatnonzero(within[i,i+1:]>=.9)+i+1:union(i,int(j))
    for index in [1,2]:
        by={}
        for i,x in enumerate(first):
            if x[index] in by:union(i,by[x[index]])
            else:by[x[index]]=i
    components=collections.defaultdict(list)
    for i in range(len(first)):components[find(i)].append(i)
    hashes={x[1] for x in earlier};rgbhashes={x[2] for x in earlier};records=[]
    for number,(_,indices) in enumerate(sorted(components.items()),1):
        direct=[bool(str(candidates.iloc[i].known_prior_SQUARE_scene).lower()=='true' or (cross[i]>=.9).any() or first[i][1] in hashes or first[i][2] in rgbhashes) for i in indices]
        keep=not any(direct)
        for i in indices:
            r=candidates.iloc[i];gid=f'confirm_refgroup_{number:04d}'
            assert gid==r.reference_group and keep==(str(r.clean_external_eligible).lower()=='true')
            assert abs(float(cross[i].max())-float(r.nearest_prior_corr))<1e-12
            records.append(dict(dataset=r.dataset,scene=r.scene,reference_group=gid,eligible=keep,nearest_prior_corr=float(cross[i].max())))
    csv(out/'classification.csv',records);print('PASS',len(records),'reference-only classifications')
if __name__=='__main__':main()
