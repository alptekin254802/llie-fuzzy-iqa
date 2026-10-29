"""Render prespecified qualitative panels from dataset pixels and recorded scores."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import peak_signal_noise_ratio
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--preprocessed',type=Path,required=True)
p.add_argument('--squarelol',type=Path,help='SQUARE-LOL root containing enhancement/')
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output=args.output.resolve();args.preprocessed=args.preprocessed.resolve()
args.output.mkdir(parents=True,exist_ok=True)
O=args.output;R=O;S=O
if args.squarelol is None:p.error('--squarelol is required for Panel B')
args.squarelol=args.squarelol.resolve()
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P))
from preprocessing.enhancement.methods import apply_clahe,apply_gamma_correction,apply_log_transform,apply_retinex,histogram_equalization
selection=json.loads((R/'selection.json').read_text())
assert selection['plan_sha256']==hashlib.sha256((P/'results/figure_protocols/rule_selection.json').read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42,'axes.unicode_minus':False})
FUNCS={'clahe':apply_clahe,'gamma':apply_gamma_correction,'log':apply_log_transform,'retinex':apply_retinex,'hist_eq':histogram_equalization}
KINDS=['low','clahe','gamma','log','retinex','hist_eq','high'];TITLES=['Low-light','CLAHE','Gamma','Log','SSR','HE','Reference']
BLUE='#0072B2';ORANGE='#D55E00';W=170/25.4*72
source_files={};image_arrays={};psnr_check=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 assert p.exists(),p
 source_files[('lol/'+str(p.relative_to(args.preprocessed)).replace('\\','/') if args.preprocessed in p.parents else 'squarelol/'+str(p.relative_to(args.squarelol)).replace('\\','/'))]=digest(p)
 a=cv2.imread(str(p));assert a is not None,p
 return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
for row in selection['lol']:
 name=row['scene'];low=load(args.preprocessed/'low'/name);high=load(args.preprocessed/'high'/name)
 assert low.shape==high.shape==(256,256,3)
 lowf=low.astype(np.float32)/255
 arrays={'low':low,'high':high}
 for rec in row['records']:
  k=rec['method'];out=np.clip(FUNCS[k](lowf),0,1)
  arrays[k]=(out*255).astype(np.uint8)
  value=peak_signal_noise_ratio(high,arrays[k],data_range=255)
  psnr_check.append({'scene':name,'method':k,'stored':rec['psnr'],'recreated':float(value),'absolute_error':abs(value-rec['psnr'])})
  assert abs(value-rec['psnr'])<2e-5,(name,k,value,rec['psnr'])
 image_arrays[name]=arrays
b=selection['squarelol'];b_arrays={}
for r in b['records']:
 im=load(args.squarelol/'enhancement'/r['name'])
 b_arrays[r['name']]=cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)

def canvas(w,h):return plt.figure(figsize=(w/72,h/72),dpi=300)
def txt(fig,w,h,x,y,s,size=8,ha='left',weight='normal',color='black'):
 return fig.text(x/w,1-y/h,s,fontsize=size,ha=ha,va='top',weight=weight,color=color)
def draw_im(fig,w,h,x,y,size,im):
 ax=fig.add_axes([x/w,1-(y+size)/h,size/w,size/h]);ax.imshow(im,interpolation='nearest');ax.set_axis_off();return ax

def panel_a(fig,w,h,rows,y0=0,title='A  LOL: scene-held-out NR-free versus negative LPIPS'):
 txt(fig,w,h,2,y0+1,title,9,weight='bold')
 txt(fig,w,h,2,y0+15,'Blue inner frame: LPIPS first',8,color=BLUE)
 txt(fig,w,h,w*.50,y0+15,'Orange outer frame: NR-free first',8,color=ORANGE)
 side=62;gap=(w-4-7*side)/6
 for i,title in enumerate(TITLES):txt(fig,w,h,2+i*(side+gap)+side/2,y0+30,title,8.5,ha='center')
 for j,row in enumerate(rows):
  y=y0+45+j*110
  txt(fig,w,h,2,y,f"{row['selection'].capitalize()} SRCC: LOL scene {row['scene_number']}   (ρ = {row['srcc']:.2f})",8.5)
  rec={r['method']:r for r in row['records']}
  for i,k in enumerate(KINDS):
   x=2+i*(side+gap);im_y=y+13
   draw_im(fig,w,h,x,im_y,side,image_arrays[row['scene']][k])
   if k not in rec:continue
   r=rec[k];text=f"PSNR {r['psnr']:.1f} dB\nLPIPS {r['lpips']:.3f}\nNR-free {r['prediction']:.1f}"
   t=txt(fig,w,h,x+side/2,im_y+side+3,text,8,ha='center');t.set_linespacing(1.05)
   for flag,color,inset in [(k in row['lpips_best'],BLUE,1.6),(k in row['nr_free_best'],ORANGE,-1.)]:
    if flag:fig.add_artist(Rectangle(((x+inset)/w,1-(im_y+side-inset)/h),(side-2*inset)/w,(side-2*inset)/h,transform=fig.transFigure,fill=False,edgecolor=color,linewidth=.8))
 return y0+45+len(rows)*110

def panel_b(fig,w,h,y):
 txt(fig,w,h,2,y,f"B  SQUARE-LOL {b['scene']}: closest to median human SRCC (ρ = {b['srcc']:.3f})",9,weight='bold')
 side=74;pitch=(w-4)/5
 for i,r in enumerate(b['records']):
  yy=y+18+(i//5)*112;x=2+(i%5)*pitch+(pitch-side)/2
  title=r['method'].replace('_',' ')
  title={'Enlightengan':'EnlightenGAN'}.get(title,title)
  txt(fig,w,h,x+side/2,yy,title,8,ha='center')
  draw_im(fig,w,h,x,yy+11,side,b_arrays[r['name']])
  text=f"Human rank {r['human_rank']:g}\nNR-free rank {r['nr_free_rank']:g}"
  t=txt(fig,w,h,x+side/2,yy+87,text,8,ha='center');t.set_linespacing(1.05)

def save(fig,p):
 fig.savefig(p,dpi=300);plt.close(fig)
# One-page candidate: physical text and image dimensions are fixed before compilation.
fig=canvas(W,620);end=panel_a(fig,W,620,selection['lol']);panel_b(fig,W,620,end+4)
save(fig,O/'fig_rule_qualitative.pdf')
info={'selection_sha256':digest(S/'selection.json'),'source_files_sha256':source_files,'method_code_sha256':digest(P/'preprocessing/enhancement/methods.py'),'PSNR_recreation_checks':psnr_check,'PSNR_max_abs_error':max(v['absolute_error'] for v in psnr_check),'LPIPS_values':'stored, no recalculation; float enhanced inputs in original pipeline','NR_free_values':'stored scene-held-out predictions for LOL, frozen LOL-trained predictions for SQUARE-LOL','display_pixels':'LOL uint8 truncation of existing float enhancement operators; SQUARE-LOL RGB area interpolation at 256x256; no further enhancement, crops or generated pixels','panel_A_image_width_bp':62,'panel_B_image_width_bp':74,'minimum_font_pt':8,'width_bp':W,'height_bp':620,'panel_B_inclusion':'included in the final supplementary figure'}
(R/'figure_evidence.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps({k:v for k,v in info.items() if k not in ['source_files_sha256','PSNR_recreation_checks']},indent=2))
