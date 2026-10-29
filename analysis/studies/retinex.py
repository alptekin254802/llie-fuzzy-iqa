"""Examine feature ranges and artifact associations for Retinex outputs."""
from analysis.studies.support import *
from analysis.studies.images import enhanced
from analysis.studies.models import fit_model, predict, serializable_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def artifacts(im, ref):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    rlab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)
    dif = lab[..., 1:] - rlab[..., 1:]
    md = dif.mean(axis=(0, 1))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(rg, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(rg, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    pos = mag[mag > 0]
    edge = (mag >= np.quantile(pos, 0.9)).astype(np.uint8) if len(pos) else np.zeros_like(mag, dtype=np.uint8)
    band = cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0
    maxr = cv2.dilate(rg, np.ones((5, 5), np.uint8))
    minr = cv2.erode(rg, np.ones((5, 5), np.uint8))
    over = np.maximum(gray - maxr, 0)
    under = np.maximum(minr - gray, 0)
    hi = gray - cv2.GaussianBlur(gray, (5, 5), 1)
    hir = rg - cv2.GaussianBlur(rg, (5, 5), 1)
    return {'chroma_mean_shift': float(np.linalg.norm(md)), 'chroma_pixel_distance': float(np.linalg.norm(dif, axis=2).mean()), 'a_mean_shift': float(md[0]), 'b_mean_shift': float(md[1]), 'brightness_signed': float(gray.mean() - rg.mean()), 'brightness_abs': float(abs(gray.mean() - rg.mean())), 'halo_overshoot': float(over[band].mean()) if band.any() else 0.0, 'halo_undershoot': float(under[band].mean()) if band.any() else 0.0, 'edge_highpass_mismatch': float(np.abs(hi[band] / (gray.std() + 1e-08) - hir[band] / (rg.std() + 1e-08)).mean()) if band.any() else 0.0, 'edge_band_pixels': int(band.sum())}

def main():
    m = begin('retinex')
    assert json.loads((OUT / 'regional_gate.json').read_text())['passed']
    d = c.lol_table()
    y = -d.lpips.to_numpy()
    train = d[d.method != 'retinex'].copy()
    test = d[d.method == 'retinex'].copy()
    yt = -train.lpips.to_numpy()
    yy = -test.lpips.to_numpy()
    result = {'distribution': {}, 'ranking': {}, 'artifacts': {}}
    any_min = np.zeros(len(test), bool)
    any_q = np.zeros(len(test), bool)
    for col in c.NR:
        lo, hi = (train[col].min(), train[col].max())
        qlo, qhi = train[col].quantile([0.05, 0.95])
        v = test[col].to_numpy()
        out = (v < lo) | (v > hi)
        outq = (v < qlo) | (v > qhi)
        any_min |= out
        any_q |= outq
        result['distribution'][col] = {'train_min': lo, 'train_max': hi, 'train_q05': qlo, 'train_q95': qhi, 'below_min_fraction': np.mean(v < lo), 'above_max_fraction': np.mean(v > hi), 'outside_minmax_fraction': out.mean(), 'below_q05_fraction': np.mean(v < qlo), 'above_q95_fraction': np.mean(v > qhi), 'outside_q05q95_fraction': outq.mean(), 'retinex_median': np.median(v)}
    result['distribution']['any_feature'] = {'outside_minmax_fraction': any_min.mean(), 'outside_q05q95_fraction': any_q.mean()}
    model = fit_model(train, yt, 'de')
    score = predict(model, d)
    result['ranking']['model'] = serializable_model(model)
    mask = d.method.to_numpy() != 'retinex'
    affine = np.linalg.lstsq(np.column_stack([np.ones(mask.sum()), score[mask]]), y[mask], rcond=None)[0]
    mapped = affine[0] + affine[1] * score
    dd = d.copy()
    dd['score'] = score
    dd['mapped_prediction'] = mapped
    dd['residual'] = mapped - y
    oof = pd.read_csv(OUT / 'validation_inputs.csv')
    result['ranking']['retinex_scene_oof_srcc'] = rho(oof.loc[oof.method == 'retinex', 'oof_nr_free'], -oof.loc[oof.method == 'retinex', 'lpips'])
    result['ranking']['method_diagnostics'] = {}
    for name, z in dd.groupby('method'):
        result['ranking']['method_diagnostics'][name] = {'evaluation': 'held_out_operator' if name == 'retinex' else 'in_sample_training_operator', 'within_method_srcc': rho(z.score, -z.lpips), 'mean_score': z.score.mean(), 'mean_negative_lpips': (-z.lpips).mean(), 'mean_residual_after_training_affine_map': z.residual.mean()}
    means = dd.groupby('method')[['score', 'lpips']].mean()
    result['ranking']['between_method_mean_srcc'] = rho(means.score, -means.lpips)
    result['ranking']['retinex_vs_rest'] = {'score_difference': score[~mask].mean() - score[mask].mean(), 'target_difference': y[~mask].mean() - y[mask].mean(), 'affine_calibrated_bias': (mapped - y)[~mask].mean()}
    dd.to_csv(OUT / 'retinex_scores.csv', index=False)
    rows = []
    manifest = {}
    for i, name in enumerate(sorted(d.filename.unique())):
        lowp = c.PREPROCESSED / 'low' / name
        refp = c.PREPROCESSED / 'high' / name
        manifest[str(lowp)] = sha(lowp)
        manifest[str(refp)] = sha(refp)
        low = c.load_rgb01(lowp, 'native')
        ref = c.load_rgb01(refp, 'native')
        for method in sorted(d.method.unique()):
            rows.append({'filename': name, 'method': method, **artifacts(enhanced(low, method), ref)})
        if (i + 1) % 100 == 0:
            print(f'Retinex artefacts {i + 1}/485', flush=True)
    aa = pd.DataFrame(rows)
    cols = [n for n in aa if n not in ('filename', 'method', 'edge_band_pixels')]
    aa = aa.merge(d, on=['filename', 'method'], validate='one_to_one')
    aa.to_csv(OUT / 'retinex_artifact_features.csv', index=False)
    result['artifacts']['method_means'] = aa.groupby('method')[cols + ['lpips']].mean().to_dict(orient='index')
    for label, z in [('pooled', aa), ('retinex_only', aa[aa.method == 'retinex']), ('scene_centered', aa[cols + ['lpips'] + c.NR] - aa.groupby('filename')[cols + ['lpips'] + c.NR].transform('mean'))]:
        result['artifacts'][label] = {a: {b: rho(z[a], z[b]) for b in ['lpips'] + c.NR} for a in cols}
    save(OUT / 'retinex_image_manifest.json', manifest)
    finish(m, result, [OUT / 'retinex_scores.csv', OUT / 'retinex_artifact_features.csv'])
if __name__ == '__main__':
    main()
