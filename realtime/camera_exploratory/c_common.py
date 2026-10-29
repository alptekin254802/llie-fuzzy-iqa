import os,sys,json,hashlib,datetime
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;BROOT=ROOT.parent;AROOT=BROOT;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(AROOT))
import cv2,numpy as np
from optimized_loop import gamma_image,optimized_candidates,select_gamma
cv2.setNumThreads(1)
PLAN=json.loads((ROOT/'PLAN.json').read_text(encoding='utf-8'))
AMENDMENT=json.loads((ROOT/'PLAN_AMENDMENT_1.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def validate_plan():
 manifest=json.loads((ROOT/'PUBLIC_EXPORT.json').read_text(encoding='utf-8'))
 for name,h in manifest['distribution_plan_sha256'].items():assert sha(ROOT/name)==h,name
 assert AMENDMENT['base_plan_sha256']==manifest['original_sha256']['PLAN.json']
 for spec in PLAN['models'].values():assert sha(REPO/spec['path'])==spec['sha256']

def models():
 validate_plan();return {k:json.loads((REPO/PLAN['models'][n]['path']).read_text()) for k,n in [('fuzzy','nr_free'),('nr_mono','nr_mono')]}
def readbacks(cap):return {p:float(cap.get(getattr(cv2,'CAP_PROP_'+p))) for p in [*PLAN['capture']['controls'],'EXPOSURE']}
def open_camera():
 validate_plan();cap=cv2.VideoCapture(PLAN['capture']['index'],cv2.CAP_DSHOW)
 if not cap.isOpened():raise RuntimeError('Could not open camera. Close Windows Camera, Teams or Zoom.')
 cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*PLAN['capture']['fourcc']));cap.set(cv2.CAP_PROP_FPS,30)
 for p,v in PLAN['capture']['controls'].items():cap.set(getattr(cv2,'CAP_PROP_'+p),v)
 cap.set(cv2.CAP_PROP_EXPOSURE,-3)
 check_settings(cap,-3)
 return cap
def check_settings(cap,exposure):
 actual=readbacks(cap);errors=[]
 for p,v in PLAN['capture']['controls'].items():
  if p=='AUTO_EXPOSURE':continue # This backend get() explicitly lacks exposure mode.
  expected=2 if p=='AUTOFOCUS' else v
  if actual[p]!=expected:errors.append(f'{p}: {actual[p]} != {expected}')
 if actual['EXPOSURE']!=exposure:errors.append(f"EXPOSURE: {actual['EXPOSURE']} != {exposure}")
 if errors:raise RuntimeError('Locked settings do not match: '+'; '.join(errors))
 return actual
def frame(cap):
 ok,im=cap.read()
 if not ok or im is None:raise RuntimeError('Could not read frame')
 if im.shape[:2]!=(480,640):raise RuntimeError('Verified resolution changed')
 return im
def settle(cap,e):
 cap.set(cv2.CAP_PROP_EXPOSURE,e)
 for _ in range(PLAN['capture']['settle_discard_frames']):frame(cap)
 return check_settings(cap,e)
def resize_input(bgr):
 return resize_rgb8(bgr).astype(np.float32)/255.0
def resize_rgb8(bgr):
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);return cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR)
def luminance8(bgr):return float(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).mean()/255)
def exposure_qc(result,baseline,stop_means,e0):
 reasons=result['reasons'][:];rules=AMENDMENT['changes']['QC'];previous=baseline;ratios=[]
 if e0-4 < -7:reasons.append('E0_minus_4_below_minus7')
 for stop,value in enumerate(stop_means,1):
  ratios.append(value/previous if previous else None)
  if value>rules['stop_max_ratio_to_previous']*previous:reasons.append('stop'+str(stop)+'_less_than_10_percent_drop')
  previous=value
 if len(stop_means)!=4:reasons.append('incomplete_dark_stops')
 elif stop_means[-1]<rules['darkest_stop_min_luminance']:reasons.append('darkest_stop_below_black_floor_margin')
 return {**result,'passed':not reasons,'reasons':reasons,'E0_minus_4':e0-4,'reference_gray8_luminance':baseline,'stop_mean_luminances':stop_means,'stop_to_previous_ratios':ratios,'darkest_stop_threshold':rules['darkest_stop_min_luminance'],'amendment_sha256':sha(ROOT/'PLAN_AMENDMENT_1.json')}
def read_reference(path):
 rgb=cv2.cvtColor(cv2.imread(str(path),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype(np.float32)/16
 return cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR)/255.0
def reference_metrics(sum_image):
 avg=sum_image.astype(np.float32)/16
 gray=cv2.cvtColor(avg,cv2.COLOR_BGR2GRAY)
 return {'mean_luminance':float(gray.mean()/255),'high_clipped_fraction':float(np.any(avg>=255,axis=2).mean())}
def qc(begin,end,settings_ok=True):
 a=begin.astype(np.float64)/(16*255);b=end.astype(np.float64)/(16*255);mse=float(np.mean((a-b)**2));psnr=float(-10*np.log10(mse)) if mse else None
 bm,em=reference_metrics(begin),reference_metrics(end);reasons=[]
 if psnr is not None and psnr<PLAN['QC']['begin_end_reference_psnr_db_min']:reasons.append('reference_drift_psnr')
 for label,m in [('begin',bm),('end',em)]:
  if not .35<=m['mean_luminance']<=.55:reasons.append(label+'_reference_luminance')
  if m['high_clipped_fraction']>=.01:reasons.append(label+'_reference_clipping')
 if not settings_ok:reasons.append('camera_settings')
 return {'passed':not reasons,'reasons':reasons,'begin_end_psnr_db':psnr,'references_identical':mse==0,'begin':bm,'end':em,'psnr_threshold_db':PLAN['QC']['begin_end_reference_psnr_db_min']}
