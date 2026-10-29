# Exploratory real-camera check
This folder contains acquisition, analysis, verification and anonymized derived tables for the exploratory webcam check.
Run in order: `python capture.py` (opens the capture GUI), then `python analyze.py --run <run>`, then `python verify.py --run <run>`.
To check the shared tables without images, use `python verify.py --derived-only --run exploratory_01`.
`export_derived.py` exports only approved tables and replaces room/view labels with `room1` and `scene_01` through `scene_21`; raw metadata stays local and unchanged.
Images and candidate caches are not shared for privacy.
The full protocol, quality checks and exclusion rules are in [Technical notes](../../docs/TECHNICAL_NOTES.md#complete-exploratory-camera-protocol).
