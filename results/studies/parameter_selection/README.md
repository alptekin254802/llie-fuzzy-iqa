# Parameter selection beyond gamma

Frozen NR-free and NR-mono were evaluated as image-only selectors. Four tasks
were screened: CLAHE clip limit, log compression, histogram-equalisation blend
and bilateral denoising strength after a fixed gamma transform. The screen used
LOL and five external subsets; it did not establish a general advantage across
tasks. CLAHE with NR-mono was selected for a subsequent frozen confirmation.

The confirmation contains 897 unused LOL-v2 synthetic images in 889 reference
groups and two real images in two groups. These are eligible pairs from the
official **training** splits, not the official benchmark test split. Previously
used scenes and overlapping normal-light references were excluded before
candidate quality was calculated. The real subset is descriptive only.

## Files

- [Screen plan](screen/PLAN.json), [summary](screen/metrics.csv),
  [paired differences](screen/contrasts.csv), [primary tests](screen/primary_tests.csv)
  and [secondary comparisons](screen/secondary_contrasts.csv).
- [Confirmation plan](confirmation/PLAN.json), [RF amendment](confirmation/PLAN_AMENDMENT_1.json),
  [summary](confirmation/metrics.csv), [paired differences](confirmation/contrasts.csv)
  and [primary tests](confirmation/primary_tests.csv).
- `group_scores.csv` in each folder contains the complete group-level inputs
  for the reported means and intervals, with no image or human-score data.
- [Model parameters](models.json), [LOL-only control choices](controls.json),
  [classical calibration rows](classical_training.csv), [fold definitions](scopes.json)
  and [RF settings](rf_settings.json). Small fuzzy and linear fits are plain JSON.
  RF weights are regenerated from the classical calibration rows, not shipped.
- [Eligibility decisions](confirmation/reference_classification.csv),
  [reference groups](confirmation/reference_groups.csv),
  [earlier references](confirmation/prior_references.csv), and the input manifests.
- [PUBLIC_EXPORT.json](PUBLIC_EXPORT.json) lists the hashes of the original plans and of the released files. File and image hashes retain full precision.

## Reproduce without images

From the repository root, with the pinned reproduction environment:

```bash
python -B -m analysis.parameter_selection.analyse --part both
python -B -m analysis.parameter_selection.verify
```

The first command recomputes all primary means, paired intervals and Holm-adjusted
tests from the group records and checks them against the archived tables.
The second also checks release hashes and exact candidate/native-output kernels.
No GPU, camera access or expensive quality inference is performed by these commands.

## Reproduce from local datasets

Copy [parameter_selection.example.json](../../../parameter_selection.example.json)
to a local configuration. Paths are relative to that configuration file. Obtain
LOL and LOL-v2 from their authors; use the exact files identified by the manifests.
For LOL, `lol_preprocessed` is the bilinear 256-square PNG preparation documented
in the main README; `lol_native` contains the original `low/` and `high/` folders.
`lolv2_archive` contains the recorded `LOL-v2/Real_captured/` and `LOL-v2/Synthetic/`
members. `lsrw` and `sice` are needed for the exploratory screen and overlap check.
Image hashes are checked before processing; substitute mirrors are not silently accepted.

```bash
python -B -m analysis.parameter_selection.classify --config local.json --output ../classification_check
python -B -m analysis.parameter_selection.run --part screen --config local.json --output ../selection_screen
python -B -m analysis.parameter_selection.run --part confirmation --config local.json --output ../selection_confirmation
```

After a complete image run, reproduce the tables from its outputs:

```bash
python -B -m analysis.parameter_selection.assemble --part screen --input ../selection_screen --output ../screen_tables
python -B -m analysis.parameter_selection.assemble --part confirmation --input ../selection_confirmation --output ../confirmation_tables
```

These steps add explicitly labelled test-set oracle controls after selection,
average within reference groups and calculate the paired comparisons. Numerical
GPU differences may affect the last digits of regenerated quality values.

The quality runs require PyTorch, pyiqa and the released LPIPS-Alex, LPIPS-VGG
and DISTS weights. They use a GPU by default (`--device cpu` is available for
reproduction, not for reproducing the reported GPU cost). Completed image rows
are retained for resumption. Work directories must be outside this repository.
The released choices and score models are frozen; rerunning does not select a
new model from confirmation outcomes.

## Numerical protocol

Input conversion is RGB, bilinear 256 x 256, float32 division by 255, followed
by the same exposure-simulation code with exposure factor 1 (no change). Feature extraction
truncates to uint8. CLAHE uses LAB lightness and an 8 x 8 grid: eight clip limits
0.5, 1, 1.5, 2, 3, 4, 6 and 8, plus bypass (-1). The default is 2 and the
LOL-trained constant is 4. Scores within 1e-9 of the maximum are tied; choose
the parameter nearest the default, then the smaller parameter. No reference is
read by the selector.

LOL screen predictions use five folds; images sharing the same or a near-identical normal-light reference stay in one fold.
External screen and confirmation predictions use the final frozen fits. Linear
and RF controls are calibrated on the five classical LOL enhancement outputs;
they are not trained on CLAHE parameter candidates. The brightness-correction
comparator is the previously frozen gamma-task model, recorded separately.

Average files within each reference group, then give groups equal weight. The
intervals use 2,000 paired group-bootstrap draws. The screen adjusts 20 external
NR-free-versus-constant tests together, with four LOL tests in a separate family.
The confirmation adjusts 12 NR-mono comparisons (six controls, two domains);
the two-image domain contributes p=1. Paired sign-flip tests use all signs for
at most ten groups and 19,999 random draws otherwise. Other intervals are
descriptive. Differences are selector minus comparator; negative LPIPS, LPIPS-VGG
or DISTS differences favour the selector, while positive PSNR/SSIM differences
favour it.

On the unused synthetic pairs, NR-mono improves LPIPS over the fixed, linear
and RF controls; LPIPS-VGG and DISTS support those comparisons. PSNR and SSIM
do not give a general NR-mono advantage. NR-free also improves over the fixed
controls, but its LPIPS difference from RF3 is unresolved. Full results, including
entropy, brightness, RF4 and test-set oracle controls, are retained above.

The exact optimisation and native webcam timings are separate records:
[runtime optimisation](../clahe_runtime/README.md) and [live timing](../clahe_live/README.md).
Quality was measured at 256 x 256; native 640 x 480 processing was timed separately.
