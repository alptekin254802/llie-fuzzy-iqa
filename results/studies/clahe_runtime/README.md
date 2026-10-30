# Exact CLAHE implementation and timing

[The optimised implementation](../../../realtime/clahe/fast.py) reuses CLAHE
objects and removes redundant image conversions. The candidate pixels, three
features, frozen model parameters and tie rule are unchanged. Archived checks
cover 1,977 images, all 17,793 candidate arrays, 53,379 feature values and 9,885
selector outputs. `FULL_RESULT.json` and `FAST_RESULT.json` retain these checks.

`primary_session*.csv` times 256 × 256 RGB images already in memory through the complete search.
`preprocessing_session*.csv` additionally includes colour conversion, bilinear
resizing and float32 preparation of resident 600 x 400 or 640 x 480 BGR inputs.
The selected output is still 256 square. These timings exclude capture, decoding,
display and native-resolution application. Three fresh sequential processes each
use 100 fixed images and three repeats, with 20 warmup calls per case. Old and
optimised cases are paired and randomly interleaved. All measured calls remain.

The `*_summary.csv` and `*_ratios.csv` files contain the session and pooled
statistics. The original `deadline_misses` column counts processing times above
1000/30 ms; it does not count dropped camera frames.

To repeat in an idle, single-thread environment, configure the local image roots
as described in [parameter selection](../parameter_selection/README.md):

```bash
python -B -m realtime.clahe.benchmark prepare --config local.json --output ../clahe_benchmark
python -B -m realtime.clahe.benchmark session --boundary primary --number 1 --config local.json --output ../clahe_benchmark
```

Repeat `session` with numbers 2 and 3 in separate sequential processes. Repeat
the three sessions for `--boundary preprocessing`, then use `aggregate` with
each boundary. These commands keep the original invocation and ordering.
Hardware and software differences can change speed; exact equality checks are
independent of the timing outcome. [Live timing](../clahe_live/README.md) adds
native-resolution output and camera acquisition.
