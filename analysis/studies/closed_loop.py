"""Evaluate gamma selection on held-out scenes with synthetic exposures."""
from analysis.studies.support_extra import *
from analysis.studies.images import basic, rgb8
from analysis.studies.models import fit_model
from evaluation.metrics import compute_psnr, compute_ssim
from preprocessing.enhancement.methods import apply_gamma_correction
GAMMAS = np.round(np.arange(0.2, 1.001, 0.1), 1)
LEVELS = [1.0, 0.25, 0.5, 0.125]

def exposed(low, level):
    linear = np.where(low <= 0.04045, low / 12.92, ((low + 0.055) / 1.055) ** 2.4) * level
    return np.clip(np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055), 0, 1).astype(np.float32)

def loadimg(folder, name):
    return c.load_rgb01(c.PREPROCESSED / folder / name, 'native')

def main():
    meta = begin2('closed_loop')
    import torch, pyiqa
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    oof = readcsv(OUT / 'validation_inputs.csv')
    foldmap = oof.groupby('filename').fold.first()
    rng = np.random.default_rng(SEED)
    selected = []
    for f in sorted(foldmap.unique()):
        selected.extend(sorted(rng.choice(sorted(foldmap[foldmap == f].index), 20, replace=False)))
    save(OUT / 'closed_loop_selection.json', {'names': selected, 'folds': {n: int(foldmap[n]) for n in selected}, 'exposures': LEVELS, 'gammas': GAMMAS, 'example_scenes': [selected[j * 20] for j in range(4)], 'data_sha256': {str(c.PREPROCESSED / d / n): sha(c.PREPROCESSED / d / n) for n in selected for d in ['low', 'high']}})
    modelsfile = OUT / 'closed_loop_models.json'
    if modelsfile.exists():
        models = json.loads(modelsfile.read_text())
    else:
        lol = c.lol_table()
        y = -lol.lpips.to_numpy()
        lf = lol.filename.map(foldmap).to_numpy()
        models = {}
        for f in sorted(foldmap.unique()):
            tr = lf != f
            models[str(f)] = {'fuzzy': fit_model(lol[tr], y[tr], 'de'), 'nr_mono': fit_product(lol[tr], y[tr]), 'training_scenes': sorted(lol.loc[tr, 'filename'].unique())}
            print('fit fold', f, flush=True)
        save(modelsfile, models)
    path = OUT / 'closed_loop_candidates.csv'
    rows = readcsv(path).to_dict('records') if path.exists() else []
    done = {(r['scene'], int(r['frame']), round(r['gamma'], 10)) for r in rows}
    dev = 'cuda'
    metrics = {m: pyiqa.create_metric(m, device=dev) for m in ['lpips', 'clipiqa+', 'niqe']}
    stopped = {}
    times = {m: [] for m in metrics}

    def ten(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(dev)
    with torch.inference_mode():
        z = ten(loadimg('low', selected[0]))
        ref = ten(loadimg('high', selected[0]))
        metrics['lpips'](z, ref)
        metrics['clipiqa+'](z)
        metrics['niqe'](z)
        torch.cuda.synchronize()
        for ix, n in enumerate(selected):
            low, high = (loadimg('low', n), loadimg('high', n))
            ht = ten(high)
            fold = int(foldmap[n])
            mods = models[str(fold)]
            for frame, level in enumerate(LEVELS):
                inp = exposed(low, level)
                for g in [*GAMMAS, 2 / 3]:
                    if (n, frame, round(g, 10)) in done:
                        continue
                    t = time.perf_counter()
                    im = np.clip(apply_gamma_correction(inp, gamma=1 / g), 0, 1).astype(np.float32)
                    genms = (time.perf_counter() - t) * 1000
                    t = time.perf_counter()
                    fe = basic(im)
                    brightness = float(cv2.cvtColor(rgb8(im), cv2.COLOR_RGB2GRAY).mean() / 255)
                    featms = (time.perf_counter() - t) * 1000
                    row = {'scene': n, 'fold': fold, 'frame': frame, 'exposure': level, 'gamma': g, 'adaptive_candidate': g != 2 / 3, 'generation_ms': genms, 'features_ms': featms, 'brightness': brightness, **fe}
                    xx = pd.DataFrame([fe])
                    for modelname, prod in [('fuzzy', False), ('nr_mono', True)]:
                        t = time.perf_counter()
                        row[modelname] = float(predict(mods[modelname], xx, prod)[0])
                        row[modelname + '_score_ms'] = (time.perf_counter() - t) * 1000
                        row[modelname + '_floor'] = bool(np.isclose(row[modelname], mods[modelname]['theta'][0], atol=1e-06, rtol=0))
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    it = ten(im)
                    torch.cuda.synchronize()
                    row['transfer_ms'] = (time.perf_counter() - t) * 1000
                    for m, fn in metrics.items():
                        if m in stopped:
                            row[m] = np.nan
                            continue
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        row[m] = float(fn(it, ht).item() if m == 'lpips' else fn(it).item())
                        torch.cuda.synchronize()
                        dt = (time.perf_counter() - t) * 1000
                        row[m + '_ms'] = dt
                        times[m].append(dt)
                        if len(times[m]) >= 10 and np.mean(times[m][-10:]) > 5000:
                            stopped[m] = 'last10 steady mean exceeded5s'
                            save(OUT / 'closed_loop_stops.json', stopped)
                    row['psnr'] = compute_psnr(im, high)
                    row['ssim'] = compute_ssim(im, high)
                    rows.append(row)
            if (ix + 1) % 5 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                save(OUT / 'closed_loop_progress.json', {'completed_scenes': ix + 1, 'total_scenes': len(selected), 'rows': len(rows), 'updated': now(), 'stopped': stopped})
                print('scenes', ix + 1, '/', len(selected), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    assert df.lpips.notna().all(), 'No valid primary endpoint'
    adaptive = df[df.adaptive_candidate.astype(bool)].copy()
    fixed = {}
    for f in sorted(foldmap.unique()):
        fixed[int(f)] = float(adaptive[adaptive.fold != f].groupby('gamma').lpips.mean().idxmin())
    names = ['fuzzy', 'nr_mono', 'entropy', 'brightness', 'niqe', 'clipiqa+', 'default', 'fixed_train', 'oracle']
    chosen = []
    for n, x in df.groupby('scene', sort=True):
        for method in names:
            if method in stopped:
                continue
            prev = 2 / 3
            for frame, a in x.groupby('frame', sort=True):
                grid = a[a.adaptive_candidate.astype(bool)].copy()
                gam = grid.gamma.to_numpy()
                oracle = float(grid.loc[grid.lpips.idxmin(), 'gamma'])
                tiecount = 1
                flat = False
                invalid = 0
                if method in ['default', 'fixed_train']:
                    g = 2 / 3 if method == 'default' else fixed[int(a.fold.iloc[0])]
                    top = a[np.isclose(a.gamma, g, rtol=0, atol=1e-12)]
                else:
                    vals = (-np.abs(grid.brightness - 0.45)).to_numpy() if method == 'brightness' else -grid.lpips.to_numpy() if method == 'oracle' else grid[method].to_numpy() * (-1 if method == 'niqe' else 1)
                    valid = np.isfinite(vals)
                    invalid = int((~valid).sum())
                    vmax = vals[valid].max() if valid.any() else np.nan
                    top = grid[valid & np.isclose(vals, vmax, atol=1e-09, rtol=0)] if valid.any() else grid
                    tiecount = len(top)
                    flat = bool(valid.all() and np.max(vals) - np.min(vals) <= 1e-09)
                choices = top.gamma.to_numpy()
                idx = np.lexsort((choices, np.round(np.abs(choices - 2 / 3), 12), np.round(np.abs(choices - prev), 12)))[0]
                r = top.iloc[idx]
                costcols = ['generation_ms']
                if method in ['fuzzy', 'nr_mono']:
                    costcols += ['features_ms', method + '_score_ms']
                elif method in ['entropy', 'brightness']:
                    costcols += ['features_ms']
                elif method in ['niqe', 'clipiqa+']:
                    costcols += [method + '_ms', 'transfer_ms']
                elif method == 'oracle':
                    costcols += ['lpips_ms', 'transfer_ms']
                runtime = float(grid[costcols].sum().sum()) if method not in ['default', 'fixed_train'] else float(r.generation_ms)
                chosen.append({'scene': n, 'fold': int(r.fold), 'frame': int(frame), 'exposure': float(r.exposure), 'selector': method, 'gamma': float(r.gamma), 'lpips': float(r.lpips), 'psnr': float(r.psnr), 'ssim': float(r.ssim), 'step_abs_dgamma': abs(float(r.gamma) - prev) if frame else np.nan, 'oracle_abs_dgamma': abs(float(r.gamma) - oracle), 'regret': float(r.lpips - grid.lpips.min()), 'uniform_tie_lpips': float(top.lpips.mean()), 'tied_max_count': tiecount, 'all_flat': flat, 'invalid_candidates': invalid, 'all_invalid_fallback': invalid == len(grid), 'selected_low_corner_floor': bool(r.get(method + '_floor', False)), 'all_candidates_low_corner_floor': bool(grid[method + '_floor'].all()) if method in ['fuzzy', 'nr_mono'] else False, 'boundary_choice': bool(np.isclose(r.gamma, 0.2) or np.isclose(r.gamma, 1)), 'selection_search_ms': runtime})
                prev = float(r.gamma)
    ch = pd.DataFrame(chosen)
    ch.to_csv(OUT / 'closed_loop.csv', index=False)
    res = {'synthetic': True, 'n_scenes': len(selected), 'fixed_train_exponent_by_fold': fixed, 'selectors': {}, 'primary_contrasts': {}, 'stopped': stopped, 'torch': torch.__version__, 'pyiqa': pyiqa.__version__, 'gpu': torch.cuda.get_device_name(0), 'pair_ties': 'half-credit exact ties; controller plateau tolerance1e-9'}
    for m, x in ch.groupby('selector'):
        res['selectors'][m] = {'mean_lpips': x.groupby('scene').lpips.mean().mean(), 'mean_psnr': x.psnr.mean(), 'mean_ssim': x.ssim.mean(), 'mean_step_abs_dgamma': x.step_abs_dgamma.mean(), 'mean_oracle_abs_dgamma': x.oracle_abs_dgamma.mean(), 'mean_regret': x.regret.mean(), 'uniform_tie_mean_lpips': x.uniform_tie_lpips.mean(), 'tied_max_frames': int((x.tied_max_count > 1).sum()), 'all_flat_frames': int(x.all_flat.sum()), 'boundary_frames': int(x.boundary_choice.sum()), 'search_runtime': stats_ms(x.selection_search_ms), 'gamma_histogram': x.gamma.value_counts().sort_index().to_dict()}
        res['selectors'][m].update(invalid_candidate_count=int(x.invalid_candidates.sum()), all_invalid_fallback_frames=int(x.all_invalid_fallback.sum()), selected_low_corner_floor_frames=int(x.selected_low_corner_floor.sum()), all_candidates_low_corner_floor_frames=int(x.all_candidates_low_corner_floor.sum()))
        if m in ['fuzzy', 'nr_mono', 'entropy', 'niqe', 'clipiqa+']:
            vals = adaptive[m].to_numpy() * (-1 if m == 'niqe' else 1)
            mask = np.isfinite(vals)
            res['selectors'][m]['candidate_rank'] = within2(vals[mask], -adaptive.loc[mask, 'lpips'], adaptive.loc[mask, 'scene'] + '_' + adaptive.loc[mask, 'frame'].astype(str))
            res['selectors'][m]['finite_candidate_rank_rows'] = int(mask.sum())
    wide = ch.groupby(['scene', 'selector']).lpips.mean().unstack()
    uw = ch.groupby(['scene', 'selector']).uniform_tie_lpips.mean().unstack()
    for b in ['fixed_train', 'entropy']:
        res['primary_contrasts']['fuzzy_minus_' + b] = diff_ci(wide.fuzzy - wide[b])
        res['primary_contrasts']['uniform_tie_fuzzy_minus_' + b] = diff_ci(uw.fuzzy - uw[b])
    finish2(meta, res, [path, OUT / 'closed_loop.csv', modelsfile])
if __name__ == '__main__':
    main()
