"""Only explicitly allowed derived tables can enter a local export archive."""
from pathlib import Path
import argparse,csv,json,zipfile,io,re
ROOT=Path(__file__).resolve().parent
ALLOWED=['candidates.csv','selections.csv','scene_means.csv','room_means.csv','exact_top1_diagnostics.csv','qc_records.json','results.json','status.json','verify.json']
IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.gif','.npy','.npz','.avi','.mp4','.mkv'}

def safe_members(source,derived_root=None):
    root=(derived_root or ROOT/'derived').resolve()
    if source.is_symlink():raise ValueError('Symlink source forbidden')
    source=source.resolve();source.relative_to(root)
    if source==root:raise ValueError('Select one derived run, not the whole derived tree')
    for p in source.rglob('*'):
        if p.is_symlink() or p.suffix.lower() in IMAGE_SUFFIXES:raise ValueError('Image/array or symlink in derived run; export refused')
    members=[]
    for name in ALLOWED:
        p=source/name
        if not p.exists():continue
        if not p.is_file() or p.is_symlink():raise ValueError('Unsafe export member')
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:header=next(csv.reader(f))
            if {'input_file','candidate_cache','image','pixels','base64'}&set(header):raise ValueError('Image/path-bearing table cannot be exported')
        members.append(p)
    if not members:raise ValueError('No derived tables to export')
    return members

def anonymized_members(source,derived_root=None):
    members=safe_members(source,derived_root)
    with (source/'candidates.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    ids=sorted({r['scene'] for r in rows});scenes={v:f'scene_{i+1:02d}' for i,v in enumerate(ids)}
    rooms={v:f'room{i+1}' for i,v in enumerate(sorted({r['room_label'] for r in rows}))}
    def anon(x,current=None):
        if isinstance(x,list):return [anon(v,current) for v in x]
        if isinstance(x,dict):
            scene=x.get('scene',x.get('scene_id',current));out={}
            for k,v in x.items():
                if k in ['started_utc','finished_utc']:continue
                if k=='direction_label':out[k]=scenes.get(scene,scene)
                elif k=='room_label':out[k]=rooms[v]
                elif k in ['scene','scene_id']:out[k]=scenes[v]
                else:out[rooms.get(k,k)]=anon(v,scene)
            return out
        if isinstance(x,str):return scenes.get(x,rooms.get(x,x))
        return x
    output={}
    for p in members:
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;records=[anon(r) for r in reader]
            stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\r\n');writer.writeheader();writer.writerows(records);text=stream.getvalue()
        else:
            data=json.loads(p.read_text(encoding='utf-8'))
            if p.name=='qc_records.json':data=[r for r in data if r.get('scene_id') in scenes]
            text=json.dumps(anon(data),indent=2,ensure_ascii=False,allow_nan=False).replace('\n','\r\n')+'\r\n'
        if re.search(r'[A-Za-z]:[\\/]',text):raise ValueError('Local absolute path in export')
        if any(original in text for original in ids):raise ValueError('Original scene identifier in export')
        output[p.name]=text.encode('utf-8')
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if Path(a.run).name!=a.run:p.error('Use a run name, not a path')
    members=anonymized_members(ROOT/'derived'/a.run)
    if a.check:print(json.dumps(list(members)));return
    if not a.output:p.error('--output required unless --check')
    dest=Path(a.output)
    if dest.suffix.lower()!='.zip':p.error('Output must be .zip')
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in members.items():archive.writestr(name,payload)
    print('Anonymized derived tables only:',dest)
if __name__=='__main__':main()
