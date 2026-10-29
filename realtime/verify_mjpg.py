"""Independent raw timestamp verification; Python standard library only."""
from pathlib import Path
import csv,json,hashlib,math,statistics,datetime
R=Path(__file__).resolve().parents[1]/'results/live_timing/mjpg_30fps'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9),(a,b)
def percentile(values,q):
    a=sorted(values);h=(len(a)-1)*q;lo=math.floor(h);hi=math.ceil(h)
    return a[lo]+(a[hi]-a[lo])*(h-lo)
def distribution(a):return {'median':statistics.median(a),'p95':percentile(a,.95),'p99':percentile(a,.99),'maximum':max(a),'mean':statistics.fmean(a)}
def verify_stats(rows,reported,new):
    assert len(rows)==reported['n']
    fields=['processing_ms','camera_interval_ms','capture_wrapper_ms','capture_to_output_ms','end_to_end_loop_ms']+(['capture_decode_ms'] if new else [])
    for field in fields:
        for key,val in distribution([r[field] for r in rows]).items():close(val,reported[field][key])
    p=[r['processing_ms'] for r in rows];c=[r['camera_interval_ms'] for r in rows];n=len(rows)
    misses=sum(x>y for x,y in zip(p,c));longs=sum(x>48 for x in c);budget=sum(x>1000/30 for x in p)
    assert misses==reported['deadline_misses'] and longs==reported['intervals_gt_48_ms'] and budget==reported['nominal_30fps_budget_misses']
    close(100*misses/n,reported['deadline_miss_percent']);close(100*longs/n,reported['interval_gt_48_ms_percent']);close(100*budget/n,reported['nominal_30fps_budget_miss_percent'])
    seconds=math.fsum(r['end_to_end_loop_ms'] for r in rows)/1000
    close(seconds,reported['measured_loop_seconds']);close(n/seconds,reported['actual_loop_fps'])
    close(statistics.median(c),reported['camera_interval_median_ms']);close(statistics.fmean(c),reported['camera_interval_mean_ms'])

def validate_csv(path,new):
    rows=list(csv.DictReader(path.open(newline='',encoding='utf-8')));assert len(rows)==2000;derived=[]
    for i,r in enumerate(rows):
        assert int(r['frame_index'])==i
        assert (r['warmup']=='True')==(i<50)
        assert float(r['gamma']) in js(R/'protocol.json')['gammas']
        t0,t1,t2=[int(r[k]) for k in ['capture_start_ns','capture_end_ns','processing_end_ns']]
        assert t0<t1<t2
        vals={'processing_ms':(t2-t1)/1e6,'capture_wrapper_ms':(t1-t0)/1e6,'capture_to_output_ms':(t2-t0)/1e6}
        close(vals['processing_ms'],float(r['processing_ms']));close(vals['capture_wrapper_ms'],float(r['capture_wait_ms']))
        if new:
            a,b=int(r['cap_read_start_ns']),int(r['cap_read_end_ns']);assert t0<=a<b<=t1
            vals['capture_decode_ms']=(b-a)/1e6;close(vals['capture_decode_ms'],float(r['capture_decode_ms']));close(vals['capture_to_output_ms'],float(r['capture_to_output_ms']))
        if i:
            prior=rows[i-1];assert t0>int(prior['processing_end_ns'])
            vals['camera_interval_ms']=(t1-int(prior['capture_end_ns']))/1e6
            vals['end_to_end_loop_ms']=(t2-int(prior['processing_end_ns']))/1e6
            close(vals['camera_interval_ms'],float(r['camera_interval_ms']))
            if new:close(vals['end_to_end_loop_ms'],float(r['end_to_end_loop_ms']))
        else:
            assert r['camera_interval_ms']==''
            if new:assert r['end_to_end_loop_ms']==''
        if i>=50:derived.append(vals)
    close(math.fsum(x['end_to_end_loop_ms'] for x in derived),(int(rows[-1]['processing_end_ns'])-int(rows[49]['processing_end_ns']))/1e6)
    return derived

def verify_archive(root):
    global R
    R=Path(root)
    result=js(R/'summary.json');plan=js(R/'protocol.json')
    for path,value in js(R/'provenance.json')['exported_file_sha256'].items():
        assert sha(R/path)==value,path
    allrows=[]
    for report in result['sessions']:
        sid=report['session'];d=R/sid;s=js(d/'status.json');test=js(d/'selftest.json')
        assert s['status']=='completed' and s['rows']==2000 and s['retained_rows']==1950
        assert test['passed'] and test['frames_completed']==100
        assert test['original_sha256']==test['lut_sha256']
        assert all(x['set_return'] for x in s['requests'])
        for stage in ['after_request_readbacks','after_settle_readbacks','after_selftest_readbacks','before_timing_readbacks','final_camera_readbacks']:
            rb=s[stage]
            assert rb['backend']=='DSHOW' and rb['fourcc_text']=='MJPG'
            assert rb['width']==640 and rb['height']==480 and abs(rb['fps']-30)<.01
            for k,v in plan['camera']['controls'].items():
                assert rb['controls'][k]==(-1 if k=='AUTO_EXPOSURE' else 2 if k=='AUTOFOCUS' else v)
            assert rb['controls']['EXPOSURE']==-6
        for label in ['before_camera','before_timing']:
            g=js(d/('load_gate_'+label+'.json'));assert g['passed'] and g['total_cpu_percent']<=10
        env=js(d/'environment.json')
        assert env['cv2_threads']==env['torch_threads']==env['torch_interop_threads']==1
        assert all(x['num_threads']==1 for x in env['threadpools'])
        data=validate_csv(d/'frames.csv',True);verify_stats(data,report,True);allrows.extend(data)
    verify_stats(allrows,result['primary'],True)
    return {'status':'PASS','retained_frames':len(allrows),'primary':result['primary']}

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=R)
    print(json.dumps(verify_archive(p.parse_args().input),indent=2))

if __name__=='__main__':main()
