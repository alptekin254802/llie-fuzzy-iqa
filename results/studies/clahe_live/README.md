# Live CLAHE selection with native-resolution output

Frozen NR-mono selects from the same nine candidates at 256 x 256. The selected
clip limit is then applied to the original 640 x 480 frame using 8 x 8 CLAHE tiles.
The native exposure/quantisation lookup table exactly matches the original
float32 implementation. The earlier [quality confirmation](../parameter_selection/README.md)
evaluates 256-square outputs; it is separate from this timing measurement.

The camera used DirectShow, MJPG, 640 x 480 at a requested 30 fps, with exposure
and controls locked as recorded in [CAMERA.json](CAMERA.json). Three rounds
rotate capture-only, gamma and CLAHE in Latin-square order. Each case starts a
fresh process and camera session, retains 2,000 frames and discards the first
50 as warmup. Display is disabled. Processing occurs on every frame with one
OpenCV/BLAS thread; no candidates or frames are skipped. No camera images,
video or audio are stored or distributed.

Each `live/round*/frames.csv` records host timestamps, processing durations and
selected parameters. The matching status files retain settings and readbacks.
The `offline/session*/` records use 100 preloaded LOL images, three repeats and
paired selection-only/native-output/gamma calls. Sessions stopped by the start-up checks have no measured frames and remain in the record; completed sessions were not replaced.

`live/summary.csv`, `offline/summary.csv`, `RESULTS.json` and `DECISION.json`
contain the session/pooled estimates and the prespecified criteria. In all three
CLAHE sessions throughput was at least 29 fps and at least 98% of capture-only
throughput, with a processing p95 below 1000/30 ms. All latency spikes remain.
Nominal-budget exceedances are processing calls longer than 1000/30 ms, not
counts of dropped frames. Host timestamps do not measure sensor-to-output latency.

Recompute the archived summaries without accessing a camera:

```bash
python -B -m realtime.clahe.summarize --output ../clahe_timing_check
```

To record a new session with a compatible camera, explicitly run:

```bash
python -B -m realtime.clahe.live --round 1 --kind capture --output ../new_camera_timing
python -B -m realtime.clahe.live --round 1 --kind gamma --output ../new_camera_timing
python -B -m realtime.clahe.live --round 1 --kind clahe --output ../new_camera_timing
```

Continue in the order in [LIVE_ORDER.json](LIVE_ORDER.json), using a fresh process
for each command. An unsupported setting or background CPU load above 10% stops
the session. Do not replace a completed session based on its result. Repeat
offline native timings with `python -B -m realtime.clahe.offline --session 1
--config local.json --output ../new_offline_timing`, then sessions 2 and 3.
