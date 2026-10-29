# Reproducing the qualitative figures

These commands draw dataset pixels and use recorded predictions. They do not fit models or generate images. Obtain LOL and SQUARE-LOL separately; prepare LOL using the bilinear pipeline in the main README. Run from the repository root with the core reproduction environment.

| Paper figure | Command | Output |
|---|---|---|
| Main Figure 2, plate-rack scene | `python -B -m analysis.qualitative.plate_rack --preprocessed preprocessed --output study_work/figures` | `fig2_qualitative_compact.pdf` |
| Supplementary rule-selected figure (Panel A: LOL; Panel B: SQUARE-LOL) | First `python -B -m analysis.qualitative.select --output study_work/figures`, then `python -B -m analysis.qualitative.rule_selected --preprocessed preprocessed --squarelol data/SQUARE-LOL --output study_work/figures` | `fig_rule_qualitative.pdf` |
| Historical two-scene qualitative figure, retained for provenance | `python -B -m analysis.make_qualitative_figure` | `results/figures/figS2_qualitative.pdf` |

Figure 2 retains scene 100.png and the same marked crop in HE and Log. Labels come from the stored PSNR, LPIPS and deployment NR-free records. No statistical scene-selection rule was recorded for this original example.

The supplementary selection rule was fixed before viewing the selected images. It chooses the highest, median and lowest scene-held-out NR-free correlation with negative LPIPS on LOL, with smallest numeric scene ID breaking ties. Panel B selects the SQUARE-LOL scene closest to median within-scene human agreement and orders its ten outputs by human score. The [selection protocol](../results/figure_protocols/rule_selection.json) records score-file hashes and tie tolerances. The selection command checks predictions against the held-out validation file before reading any images.

LOL display pixels reproduce the original enhancement operations; PSNR is checked against stored values, while LPIPS and NR-free labels are read from the recorded float-input and held-out evaluations. SQUARE-LOL uses area interpolation to 256 square. Labels remain at least 8 pt at the saved physical figure size. Evidence JSON files record displayed values and image-source hashes. Raw dataset images are not distributed here.
