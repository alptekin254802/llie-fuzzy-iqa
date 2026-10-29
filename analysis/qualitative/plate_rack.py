"""Redraw recorded figures at their final physical size; no fitting or new measurements."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator, FuncFormatter

import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--preprocessed',type=Path,required=True)
p.add_argument('--squarelol',type=Path,help='SQUARE-LOL root containing enhancement/')
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output=args.output.resolve();args.preprocessed=args.preprocessed.resolve()
args.output.mkdir(parents=True,exist_ok=True)
O=args.output
R=Path(__file__).resolve().parents[2]/'results'
I=args.preprocessed
sys.path.insert(0,str(R.parent))
from preprocessing.enhancement.methods import apply_log_transform, histogram_equalization
from evaluation.frozen_scorer import FrozenScorer

# PDF big points (1/72 inch), matched to the unchanged svjour3 / supplement widths.
MAIN=493.15; COL=238.08; HALF=.48*MAIN; SUP=459.21
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.labelsize':8.5,
                    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                    'axes.spines.right':False,'axes.unicode_minus':False})
C={'blue':'#4c72b0','red':'#c44e52','orange':'#dd8452','purple':'#8172b3','green':'#55a868'}
evidence={}; sizes={}
def canvas(w,h):return plt.figure(figsize=(w/72,h/72),dpi=300)
def save(fig,name,folder='manuscript'):
    # Fixed canvas: never let tight bounding-box cropping silently shrink final fonts.
    fig.savefig(O/(name+'.pdf'),dpi=300)
    sizes[folder+'/'+name+'.pdf']={'width_bp':fig.get_figwidth()*72,'height_bp':fig.get_figheight()*72}
    plt.close(fig)
def read(n):return pd.read_csv(R/n)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# Same plate-rack scene, unmodified enhancement operators, and recorded score labels.
mt=read('enhancement_metrics.csv')[['filename','method','psnr']].merge(
    read('perceptual_metrics.csv')[['filename','method','lpips']],
    on=['filename','method'],validate='one_to_one').merge(
    read('nr_fuzzy_scores.csv')[['filename','method','nr_fuzzy_free']],
    on=['filename','method'],validate='one_to_one').set_index(['filename','method'])
scene='100.png'
low=cv2.cvtColor(cv2.imread(str(I/'low'/scene)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
high=cv2.cvtColor(cv2.imread(str(I/'high'/scene)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
ims=[low,np.clip(histogram_equalization(low),0,1),np.clip(apply_log_transform(low),0,1),high]
kinds=['low','hist_eq','log','high']; titles=['Low-light','Hist. Eq.','Log','Reference']
fig=canvas(MAIN,164); records=[]
panel=115; gap=(MAIN-4*panel-4)/3
roi=(184,62,232,110)
for col,(kind,im,title) in enumerate(zip(kinds,ims,titles)):
    x=2+col*(panel+gap)
    ax=fig.add_axes([x/MAIN,29/164,panel/MAIN,panel/164])
    ax.imshow(im,interpolation='nearest');ax.set_axis_off()
    fig.text((x+panel/2)/MAIN,153/164,title,fontsize=9.5,ha='center',va='center')
    if kind in ('hist_eq','log'):
        row=mt.loc[(scene,kind)]
        fig.text((x+panel/2)/MAIN,1.5/164,f'PSNR {row.psnr:.1f} dB\nLPIPS {row.lpips:.2f}\nNR-free {row.nr_fuzzy_free:.0f}',
                 fontsize=9,ha='center',va='bottom',linespacing=1.0)
        x0,y0,x1,y1=roi
        ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#ffdb40',linewidth=.9))
        inset=fig.add_axes([(x+panel-55)/MAIN,31/164,53/MAIN,53/164])
        inset.imshow(im[y0:y1,x0:x1],interpolation='nearest')
        inset.set_xticks([]);inset.set_yticks([])
        for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#ffdb40');spine.set_linewidth(.9)
        records.append({'scene':scene,'method':kind,'psnr':float(row.psnr),'lpips':float(row.lpips),'nr_free':float(row.nr_fuzzy_free)})
save(fig,'fig2_qualitative_compact')
evidence['qualitative']={'records':records,'scene':scene,'roi_xyxy':roi,'crop_interpolation':'nearest',
                         'crop_processing':'none; identical pixel region in Hist. Eq. and Log','labels_pt':9}


(O/"figure2_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
