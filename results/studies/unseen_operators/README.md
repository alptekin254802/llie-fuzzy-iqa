# Exploratory transfer to unseen enhancement methods

This check, added during revision, uses the same 485 LOL calibration scenes and evaluates eight additional enhancement methods. It is not a held-out-scene experiment. Table S15a reports the compact comparison; all intervals and single-feature results remain in `results.csv`.

## Records and verification

- `predictions_*.csv`: per-scene features, LPIPS and three score columns, with no images or human labels.
- `classical_pool_model.json`: five-method calibration; `four_classical_model_posthoc.json`: the later analysis excluding SSR-50.
- `results.csv`: complete results plus explicitly labelled historical and skipped rows. Four-method values, `brightness_srcc_posthoc` and `neg_noise_srcc_posthoc` are post-hoc; the original five-method values are unchanged.
- [Protocol](PLAN.md), [source hashes](protocol_sources.json), [upstream assets](external_sources.json) and [licence check](license_check.json).

From the repository root, the following command recomputes every displayed correlation and confidence interval from the included numeric records, checks frozen-model scores and the SSR-50 control, and writes only the optional report:

```bash
python -B -m analysis.unseen_operators verify --output /path/to/new_verification.json
```

To recreate measurements, first prepare LOL using the main README. Supply the resulting `low/` and `high/` folders, each containing all 485 paired 256-square inputs. No further resizing occurs. The command accepts one method at a time and refuses to overwrite its output:

```bash
python -B -m analysis.unseen_operators measure --method MSRCR --data-root /path/to/preprocessed --external-root /path/to/external --output /path/to/new_msrcr.csv
python -B -m analysis.unseen_operators measure --method AGCWD --data-root /path/to/preprocessed --external-root /path/to/external --output /path/to/new_agcwd.csv
python -B -m analysis.unseen_operators measure --method Zero-DCE++ --data-root /path/to/preprocessed --external-root /path/to/external --output /path/to/new_zero_dce_pp.csv
python -B -m analysis.unseen_operators measure --method RUAS-MIT --data-root /path/to/preprocessed --external-root /path/to/external --output /path/to/new_ruas_mit.csv
```

SSR-15, SSR-80, SSR-250, SSR-50 and MSR are also accepted. Optional `calibrate --pool 4|5 --output /path/to/new_model.json` refits the archived calibration settings from the included classical-feature records. No fitting is needed for the verification command.

## Obtain upstream implementations and weights separately

Download each file from its pinned `url` in `external_sources.json`, preserving its corresponding `local` path beneath the external directory. The adapter checks SHA-256 before loading external code or weights. Required layout:

| Source | Local files beneath external directory |
|---|---|
| [Retinex](https://github.com/dongb5/Retinex/tree/e283325c0a3c6ce12607aa67ad68082bf5327608) | `retinex/retinex.py`, `retinex/config.json` |
| [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension/tree/09f202b690f82da939b8e6ec8535960ae97ad8bd/Zero-DCE%2B%2B) | `zero_dce_pp/model.py`, `zero_dce_pp/Epoch99.pth` (upstream checkpoint is under `snapshots_Zero_DCE++/`) |
| [RUAS](https://github.com/vis-opt-group/RUAS/tree/461172c4d05b54155d8f818af0543eca457a1bb5) | `ruas/model_enhance/model.py`, `operations.py`, `genotypes.py` in the same subfolder, and `ruas/weights/mit.pt` |

Use the MIT5K RUAS checkpoint, not its LOL checkpoint. Zero-DCE++ uses the official SICE Part1 checkpoint. No images, external implementations or weights are redistributed here. At the pinned revisions, the Zero-DCE README specifies academic research use under CC BY-NC 4.0; no explicit redistribution licence was found for Retinex or RUAS. Consult the upstream terms before obtaining these assets. Our adapter and independently implemented MSR/AGCWD routines remain in `analysis/unseen_operators.py`.

See [Technical notes, S10](../../../docs/TECHNICAL_NOTES.md#exploratory-unseen-enhancement-check) for the analysis chronology and limitations.
