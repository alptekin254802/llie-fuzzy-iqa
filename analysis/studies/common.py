"""Shared feature transforms, fuzzy calibration and ranking statistics."""
from __future__ import annotations
import itertools, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import kendalltau, pearsonr, spearmanr
from analysis.studies.config import *
from evaluation.nr_features import nr_features as repo_nr_features
NR = ['entropy', 'contrast', 'sharpness']
ALL6 = ['brightness', 'contrast', 'entropy', 'colorfulness', 'sharpness', 'noise']
K_FOLDS, DE_MAXITER, DE_POPSIZE, SEED = (5, 40, 8, 42)
COMBOS = list(itertools.product(range(3), range(3), range(3)))

def load_rgb01(path, mode='area256'):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mode == 'area256':
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    elif mode == 'linear256':
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    elif mode != 'native':
        raise ValueError(mode)
    return rgb.astype(np.float32) / 255.0

def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)

def mfset(v):
    q = np.quantile(v, [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]

def srho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)

def firing(arrs, mfs):
    Ms = [np.stack([trapmf(a, *mfs[n][l]) for l in range(3)], axis=1) for n, a in enumerate(arrs)]
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k]) for i, j, k in COMBOS], axis=1)
    return (fire, fire.sum(axis=1) + 1e-09)

def calibrate(fire, fsum, target, idx, seed=SEED):

    def obj(t):
        s = fire[idx] @ t / fsum[idx]
        return -srho(s, target[idx])
    return differential_evolution(obj, bounds=[(0.0, 100.0)] * 27, maxiter=DE_MAXITER, popsize=DE_POPSIZE, seed=seed, tol=0.001, polish=False, workers=1, updating='deferred').x

def fit_signs(X, target, cols=NR):
    return {c: bool(spearmanr(X[c].to_numpy(float), target)[0] < 0) for c in cols}

def oriented(X, signs, cols=NR):
    return [-X[c].to_numpy(float) if signs[c] else X[c].to_numpy(float) for c in cols]

def group_kfold(groups, k=K_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(groups)))
    rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist())
        m = np.array([g in ts for g in groups])
        yield (np.where(~m)[0], np.where(m)[0])

def _logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5

def iqa_scores(pred, mos):
    pred = np.asarray(pred, float)
    mos = np.asarray(mos, float)
    srcc = srho(pred, mos)
    krcc = kendalltau(pred, mos)[0]
    try:
        p0 = [np.max(mos) - np.min(mos), 1.0 / (np.std(pred) + 1e-06), float(np.mean(pred)), 0.0, float(np.mean(mos))]
        b, _ = curve_fit(_logistic5, pred, mos, p0=p0, maxfev=20000)
        plcc = pearsonr(_logistic5(pred, *b), mos)[0]
    except Exception:
        plcc = pearsonr(pred, mos)[0]
    return (float(srcc), float(krcc), float(plcc))

def within_group(pred,target,groups):
    data=pd.DataFrame({'p':np.asarray(pred),'t':np.asarray(target),'g':np.asarray(groups)})
    sr,kr,top=[],[],[];correct=0.;pairs=0
    for _,z in data.groupby('g',sort=True):
        if len(z)<2 or z.t.nunique()<2:continue
        p=z.p.to_numpy();y=z.t.to_numpy();sr.append(srho(p,y));kr.append(kendalltau(p,y).statistic)
        selected=p==p.max();top.append(float(np.mean(y[selected]==y.max())))
        i,j=np.triu_indices(len(z),1);valid=y[i]!=y[j];a=np.sign(p[i]-p[j]);b=np.sign(y[i]-y[j])
        correct+=float(np.sum((a==b)&valid)+.5*np.sum((a==0)&valid));pairs+=int(valid.sum())
    return {'groups':len(sr),'mean_srcc':float(np.mean(sr)),'mean_krcc':float(np.nanmean(kr)),'top1':float(np.mean(top)),'pair_acc':correct/max(pairs,1),'pairs':pairs}

def lol_table():
    nf = pd.read_csv(RES / 'nr_features.csv')
    pc = pd.read_csv(RES / 'perceptual_metrics.csv')[['filename', 'method', 'lpips', 'niqe', 'brisque']]
    em = pd.read_csv(RES / 'enhancement_metrics.csv')[['filename', 'method', 'psnr', 'ssim']]
    df = nf.merge(pc, on=['filename', 'method']).merge(em, on=['filename', 'method']).dropna().reset_index(drop=True)
    return df

def lol_frozen_model():
    from evaluation.frozen_scorer import FrozenScorer
    p = FrozenScorer().parameters
    return (p['signs'], p['mfs'], np.array(p['theta']))

def fuzzy_predict(X, signs, mfs, theta, cols=NR):
    fire, fsum = firing(oriented(X, signs, cols), mfs)
    return fire @ theta / fsum
