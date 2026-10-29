# Zero-shot Q-Align comparison

This additional experiment evaluates Q-Align with fp16 weights using the official
[q-future/one-align checkpoint](https://huggingface.co/q-future/one-align), revision
`dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb`. The model is described in
[Wu et al., ICML 2024](https://proceedings.mlr.press/v235/wu24ah.html).
No LLIE training, MOS calibration, prompt selection, sign selection or checkpoint
selection is performed. Accuracy tables use fp16 scores; latency remains the
recorded 4-bit NF4 / float16-compute GPU measurement. The checkpoint, prompt,
processor and five quality-level token weights are identical.

## Environment and inputs

Use a separate Python 3.11 environment. The recorded run used PyTorch 2.7.1+cu118,
torchvision 0.22.1+cu118 and the versions in `requirements-qalign.txt`. Install the
CUDA 11.8 PyTorch wheels for a compatible system, then those requirements. Use a separate environment to avoid changing dependencies of the other studies.

Configure the three human datasets as described in [STUDIES.md](STUDIES.md).
Run `features DATASET area256 36000` for `squarelol`, `lieq` and `rlie` first.
The fixed sets contain 2,900, 1,000 and 1,540 enhanced images in 290, 100 and 154
scenes. Each original file is decoded by OpenCV, converted BGR to RGB, resized
with INTER_AREA to 256 x 256, and converted to a float32 [0,1] tensor. Q-Align's
standard processor subsequently resizes/crops to 448 x 448 and normalizes it.
Its fixed image-quality prompt and probability-weighted score are unchanged;
higher scores mean better quality. LOL is not evaluated by this experiment.

```bash
python -B -m analysis.studies --config study_config.local.json qalign prepare
python -B -m analysis.studies --config study_config.local.json qalign run
python -B -m analysis.studies --config study_config.local.json qalign analyze
```

The original 4-bit runner uses a dedicated checkpoint cache and pins its main reference to the
recorded revision. Downloading, loading and the first valid score have a two-hour
setup deadline. There is no inference-speed cutoff after setup succeeds. Each
image is checkpointed, so the same command can resume an interrupted run. A
complete run is required for analysis. No CPU/disk offload or silent fp16 fallback
is accepted. The `prepare` command recomputes NR-free and, when the recorded
baseline CSV is present, validates it before preserving the archived prediction
precision. It does not fit either model.

## Accuracy and the retained 4-bit timing

The primary outcome is mean within-scene SRCC. Secondary outcomes are pooled
SRCC, within-scene pair accuracy and top-1 agreement. Average ranks are used; constant predictions
receive zero scene SRCC. Target-tied pairs are skipped and prediction ties receive
one-half credit. Top-1 ties use shared per-image random keys (seed 20261021); the uniform-tie
expectation is also recorded. All outcomes are reported for all three datasets.

Q-Align minus NR-free uses paired whole-scene bootstrap resampling: 2,000 draws,
seed 20261018, percentile 95% intervals and null-centered two-sided p-values with
a plus-one correction. These additional p-values are unadjusted. The sign of the
difference is explicit: positive favours Q-Align. Raw and per-scene outputs are
saved alongside the JSON summaries.

The retained 4-bit timing uses a GTX 1060 6 GB under Windows WDDM, batch size one and one
CPU thread. Nine pilot calls (the first three filenames of each dataset) precede
and are excluded from the measured calls. CUDA synchronization brackets scoring.
The primary latency includes Q-Align's internal processor and forward call on an
in-memory AREA256 tensor. File read/external AREA preparation and the first host
to device transfer are recorded separately. Median, 95th percentile and mean are
reported without dropping slow calls. An interval of concurrent desktop GPU
activity occurred during RLIE scoring and is retained. These are observed desktop
latencies, separate from the paper's single-thread CPU comparison.

## Reproduce fp16 accuracy

After preparing the same feature tables and manifest, recompute all fp16
accuracy metrics and paired contrasts without loading a model:

```bash
python -B analysis/qalign_fp16/analyze.py --manifest study_work/qalign_manifest.csv --fp16-scores results/studies/qalign_fp16_scores.csv --fourbit-scores results/studies/qalign_scores.csv --fourbit-results results/studies/qalign_results.json --output study_work/qalign_fp16_analysis
```

The manifest must retain the full-precision frozen NR-free predictions and
human scores. Use `qalign prepare` above with the same dataset labels. CSVs
are read with round-trip floating-point precision to retain ties exactly.
`publication_results.json` contains fp16 accuracy and paired NR-free contrasts
with **4-bit** timing copied unchanged from its named source;
`precision_results.json` compares both precisions and includes top-1 accuracy.
The shipped counterparts are under `results/studies/qalign_fp16_*`.
Images and opinion labels remain external and must be obtained from their authors.

To rescore the same inputs in fp16, use a separate Python 3.11 environment with
`requirements-qalign-fp16.txt` (and CUDA 11.8 PyTorch wheels as above):

```bash
python -B analysis/qalign_fp16/run.py --manifest study_work/qalign_manifest.csv --fourbit-scores results/studies/qalign_scores.csv --fourbit-environment results/studies/qalign_environment.json --output study_work/qalign_fp16_run --cache-dir study_work/qalign_fp16_cache
```

The portable runner retains the recorded fp16 loader, exact external preprocessing
and unchanged installed Q-Align scoring code. It verifies checkpoint shard hashes,
implementation hashes and input RGB hashes against the 4-bit run. The model uses
`torch.float16`, `device_map="auto"` and a 5 GiB GPU budget; remaining weights are
offloaded to CPU. It refuses quantized layers and disk-offloaded weights. Free
physical RAM must be at least 16 GiB before loading. Run only one worker per
output directory; the lock directory is removed on normal exit.

A seeded 50-image SQUARE-LOL pilot precedes the full run. A prediction is flushed
and synced to CSV after every image; rerunning resumes validated completed rows.
NaN/inf stops the run. If the pilot estimates more than 14 hours, the runner limits
the run to SQUARE-LOL and RLIE; the archived campaign completed all three sets.
The fixed order is SQUARE-LOL, RLIE, LIEQ. Use the generated `scores.csv` as the
`--fp16-scores` argument to the analyzer. Do not mix precisions in one CSV.

**fp16 CPU-offload elapsed time is a feasibility measurement, not Q-Align's
reported computational cost.** The paper's 1.18 s/image remains the 4-bit GPU
measurement. The original `qalign_scores.csv`, `qalign_results.json` and original
4-bit reproduction commands above remain available for timing and provenance.

The original comparison plan SHA-256 is
`8813b8271d67b093a3ce3891b6ab8e8f98f386c6cc4f930b165df2fad705775a`.
The hash identifies the original plan, not this reproduction guide.

The fp16 verification plan SHA-256 is
`feeb4f9907eeb8b508fbb4f754125d632944b64e046a1fe3f310344e91c195bd`.

| Dataset | Within SRCC | Pooled SRCC | Pair accuracy | Top-1 |
|---|---:|---:|---:|---:|
| SQUARE-LOL | 0.473 | 0.180 | 0.679 | 0.131 |
| LIEQ | 0.575 | 0.825 | 0.725 | 0.200 |
| RLIE | 0.416 | 0.244 | 0.657 | 0.130 |
