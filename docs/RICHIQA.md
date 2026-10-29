# RichIQA zero-shot comparison

The released author checkpoint is evaluated without LLIE training, label calibration,
checkpoint selection or sign selection. Predictions use the unmodified official
`inference.py` from [YixuanGao98/RichIQA](https://github.com/YixuanGao98/RichIQA),
commit `bfcf34939937e8fd2a55179982d648c4dfb29c37`.

## Weights and environment

The [Hugging Face release](https://huggingface.co/GYX98/RichIQA) is pinned to
`af8ff09037533c265c8a57d981319b549b253a4b`. `best.pkl` SHA-256 is
`22fda9c41ef3b03f71c828565056dd92993a30a299b93aab120ce4533f5ebdb7`.
The constructor also loads `CvT-21-384x384-IN-22k.pth`; obtain that exact file from
the README's [official OneDrive model zoo](https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al).
Its SHA-256 is `e630d7b77ac3777832a1d4c89e989dd2b51459d6eebe8f96816e4d794f51a14f`.
Set `richiqa_backbone` in the local study configuration to its absolute path.
The HF repository contains the final RichIQA checkpoint, not this separate initial backbone.
The strict final checkpoint supplies all network parameters, including the backbone.

Use Python 3.11, the normal study dependencies and the tested versions recorded
in `requirements-richiqa.txt`. Install CUDA 11.8 builds of torch/torchvision separately
for the recorded GTX 1060 configuration. Inference is float32, batch one. No source
patch, alternate head, quantization, or CPU offload is used. The official constructor
requires CUDA even when loading the model for the subsequent CPU benchmark.

## Data and commands

Configure LOL, its bilinear-preprocessed images and the three human datasets as in
[STUDIES.md](STUDIES.md). Prepare human `area256` feature tables first. The command
below materializes the same 8-bit images used by the other additional metrics and
verifies all decoded RGB pixel hashes against the archived experiment. LOL contains
2,425 outputs from 485 scenes; SQUARE-LOL 2,900/290, LIEQ 1,000/100, RLIE 1,540/154.
LOL's existing enhanced outputs are already 256-square, so an outer AREA resize is
identity; human inputs use INTER_AREA. The official inference function then applies
PIL bilinear resize to 224-square, ImageNet normalization and its own model call.
Its scalar is the expected score over categories 1..5; higher MOS is better.
The README describes mean-quality prediction but does not explicitly give score direction.

```bash
python -B -m analysis.studies --config study_config.local.json richiqa setup
python -B -m analysis.studies --config study_config.local.json richiqa prepare
python -B -m analysis.studies --config study_config.local.json richiqa run
python -B -m analysis.studies --config study_config.local.json richiqa analyze
```

`prepare` reuses the shared feature/frozen-model helper also used by Q-Align; it
neither loads Q-Align nor fits any model. The original experiment fixed a two-hour
setup cap and succeeded within that limit. The wrapper checks complete inputs,
source/weight hashes, finite predictions and strict checkpoint equality. Scores are
checkpointed per image; incomplete CPU timing files require diagnosis before resuming.
Do not mix scores produced with different input or weight hashes.

## Measures and timing

LOL reports pooled, within-scene and five-fold mean SRCC against negative LPIPS;
NR-free uses its original out-of-fold predictions. Human datasets report within-scene
and pooled SRCC plus pair accuracy, compared with the frozen NR-free deployment model.
Ranks are averaged; constant predictions receive zero scene SRCC, constant targets
are excluded from within-scene correlation, target-tied pairs are skipped and
prediction ties receive half credit.

Paired scene bootstrap uses 2,000 draws and seed 20261019, percentile 95% intervals
and null-centered two-sided plus-one p-values. These added p-values are unadjusted.
LOL fold-mean contrasts resample scenes within each fixed fold. No refitting occurs.
The additionally saved LOL pair-accuracy field is explicitly exploratory and is not
used in the manuscript's RichIQA outcome tables.

Both devices use one CPU thread. The official full prediction call is timed on
prepared PNG inputs, including decode, internal preprocessing, transfer, forward
pass and scalar conversion. External AREA preparation and model loading are excluded.
CPU uses the first 25 sorted filenames from each dataset, in three consecutive
100-image sessions. GPU uses all images with CUDA synchronization. Warm-ups are
excluded; all subsequent calls, including slow ones, are retained. The CPU module
is passed to the unchanged official prediction function, and CPU/GPU agreement is checked.

## Archived-score reproduction

After `prepare`, copy the archived `richiqa_scores.csv`, `richiqa_cpu.csv`,
`richiqa_environment.json`, `richiqa_run.json` and `richiqa_sources.json` from
`results/studies/` to the configured work directory, then run `richiqa analyze`.
This recalculates all metrics and intervals without rerunning the neural network.
Archived score files contain identifiers, predictions and timings, but no images,
human opinion labels or private filesystem paths. A new GPU run also requires `setup`;
do not use the metadata-only archived source record to load weights.

Original pre-result plan SHA-256:
`410be20fbfd118b83ddc451855c993ded41d44fb9d82ac421facf3c53eb66dac`.
The hash identifies the original plan, not this reproduction guide.
