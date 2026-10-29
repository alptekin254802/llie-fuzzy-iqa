# Exact LUT candidate search for a webcam

Run these commands from the repository root. No model training, network downloads or GPU are required. A physical Windows DSHOW webcam is required for live use and the default selftest. The default camera profile was tested with a USB Camera at 640x480 MJPG; other devices require an independently validated manual-control profile. Requested 30 fps is not assumed to be the delivered frame rate.

Use Python 3.11 with `realtime/requirements.txt`. Install the full `opencv-python` package, not `opencv-python-headless`, for DSHOW support. Use a dedicated environment; do not install both variants together.

```powershell
python -m pip install -r realtime/requirements.txt
python -B realtime/selftest.py --models models --frames 100
python -B realtime/live_loop.py --models models --exposure -6 --frames 2000 --warmup 50 --output live_run_01
```

Close Windows Camera, Teams and other camera applications before opening the webcam. Keep the camera, illumination and scene fixed. Every live launch compares 100 actual webcam frames against an independent floating reference before timing. Failure aborts the run. No preview is shown and no scene images are stored. The first 50 of the 2000 total frames are flagged as warmup and excluded from summary statistics.

The environment variables OMP/MKL/OPENBLAS/NUMEXPR are set to one before numerical imports; OpenCV and threadpoolctl also enforce one thread. `--external-state path.json` can block measurement while another experiment reports `status=running`; it never changes or stops that experiment.

`kernel.py` is the measured path: BGR to RGB, uint8 bilinear 256x256 resize, nine lookup-table candidates, entropy/contrast/sharpness, batched NR-free prediction, original gamma tie rule, and full-resolution LUT correction. Model/table construction, camera wait, image display and CSV serialization are excluded from processing latency. The full loop fps includes capture and between-frame work.

The table uses exactly the floating reference operations on each of the 256 input values. Grid gamma values remain NumPy float64; the exponent is `1.0/(1/g)`, with the original clips, float32 casts and RGB8 conversion. Default 2/3 stays a Python float and is tested separately. Do not simplify the reciprocal exponent or change scalar types. The optimized batch contraction preserves the original per-row reduction order.

Selftest checks RGB8 candidate bytes, all three features, brightness, both NR models and the four selector decisions. It uses the same ties, previous-gamma continuity, default proximity and lower-gamma fallback as the study. `reference.py` is an independent copy of the original operations, not the optimized kernel. For an offline installation check without a camera:

```powershell
python -B realtime/selftest.py --models models --synthetic --frames 100
```

The synthetic quality study retains floating-point candidates. This live pipeline measures processing latency. The separate [exploratory camera check](camera_exploratory/README.md) compares image quality on paired exposures.

Camera interval is the spacing between completed host reads, not a hardware sensor timestamp. Deadline miss is processing time greater than that frame's observed interval; back-pressure can increase the interval, so this statistic alone is not a hard real-time guarantee. All percentiles use NumPy's default linear quantile without removing outliers.

## Recompute the recorded timing results

`results/live_timing/mjpg_30fps/` contains three MJPG sessions at exposure -6: 2,000 frames per session, with the first 50 excluded, leaving 5,850 measured frames. The launcher sets dimensions and FPS before FOURCC and rejects a non-MJPG readback.

```powershell
python -B -m realtime.summarize
```

The independent standard-library verifier checks hashes, raw timestamp identities, camera readbacks, selftests and single-thread settings. The pooled processing median is 8.58 ms and the maximum is 22.53 ms; capture-inclusive throughput is 29.55 frames/s. Processing includes nine candidates, scoring, selection and full-resolution correction. No image is included.

`protocol.json`, `summary.json` and `provenance.json` describe the protocol, numeric results and original/distribution hashes. Environment and load checks accompany each session. Local paths and process identities are omitted from distribution metadata; frame CSVs and selftests are byte-identical to their originals.

Earlier sessions remain under `results/live_timing/` and can be checked with `python -B -m realtime.summarize --input results/live_timing`. Their primary exposure -5 cohort has median 9.2632 ms and throughput 25.019 frames/s. Its YUY2 format was inferred retrospectively, not recorded in the original sessions. Both exposure and format changed, so the difference cannot be attributed solely to MJPG.

The separate floating-point synthetic repeat is archived in
`results/synthetic_timing/`. To verify its component summaries and all 1,600
gamma selections, run `python -B -m realtime.summarize_synthetic`. The median
sum of generation and NR-free scoring components changes from 82.25275 to
67.99570 ms. These component sums use different inputs and timing boundaries
from the live-camera measurement above. The raw timing/profile CSVs and the
recorded equality result are included; the source images remain external.
