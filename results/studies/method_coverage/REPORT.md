# Training-method coverage

The experiment retains the S14 256x256 area-interpolated features, five stored scene folds, frozen LOL orientations and memberships, ridge=0.001 fuzzy least squares, and scene-centred linear regression. The preparation check reproduces the three S14 models on all three datasets to three decimals.
Each of 20 repeats (seed 20261029) holds four test methods fixed across nested training sets of two to six methods. Training image counts increase with method counts. The LOL prior uses no human-score fitting.
Values below are means and population SDs over 20 method repeats x 5 scene folds.

| Dataset | Model | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---:|---:|---:|---:|---:|
| SQUARE-LOL | fuzzy LS | 0.005 +/- 0.226 | -0.078 +/- 0.246 | -0.082 +/- 0.230 | -0.012 +/- 0.226 | 0.012 +/- 0.208 |
| SQUARE-LOL | linear-3f | -0.019 +/- 0.297 | -0.102 +/- 0.285 | -0.044 +/- 0.302 | 0.047 +/- 0.333 | 0.072 +/- 0.330 |
| SQUARE-LOL | LOL prior | 0.267 +/- 0.212 | 0.267 +/- 0.212 | 0.267 +/- 0.212 | 0.267 +/- 0.212 | 0.267 +/- 0.212 |
| LIEQ | fuzzy LS | 0.041 +/- 0.236 | 0.046 +/- 0.206 | 0.052 +/- 0.190 | 0.077 +/- 0.161 | 0.130 +/- 0.163 |
| LIEQ | linear-3f | -0.018 +/- 0.290 | 0.032 +/- 0.283 | 0.040 +/- 0.238 | 0.099 +/- 0.216 | 0.168 +/- 0.240 |
| LIEQ | LOL prior | 0.052 +/- 0.174 | 0.052 +/- 0.174 | 0.052 +/- 0.174 | 0.052 +/- 0.174 | 0.052 +/- 0.174 |
| RLIE | fuzzy LS | 0.199 +/- 0.140 | 0.231 +/- 0.133 | 0.249 +/- 0.128 | 0.274 +/- 0.115 | 0.293 +/- 0.108 |
| RLIE | linear-3f | 0.310 +/- 0.140 | 0.347 +/- 0.124 | 0.359 +/- 0.124 | 0.372 +/- 0.112 | 0.378 +/- 0.118 |
| RLIE | LOL prior | 0.069 +/- 0.142 | 0.069 +/- 0.142 | 0.069 +/- 0.142 | 0.069 +/- 0.142 | 0.069 +/- 0.142 |

| Fuzzy LS: k=6 minus k=2 | Difference | 95% CI |
|---|---:|---:|
| SQUARE-LOL | +0.007 | [-0.011, +0.024] |
| LIEQ | +0.088 | [+0.040, +0.135] |
| RLIE | +0.095 | [+0.066, +0.122] |

Intervals use 2,000 paired whole-scene bootstrap draws (seed 20261029). All 20 repeated observations of a scene move together. Fixed scene folds retain equal weight. Fits and method splits stay fixed; intervals are marginal and unadjusted.
Differences are the six-method fit minus the two-method fit; positive values favour the six-method fit. Secondary contrasts compare each human-calibrated fit with the LOL prior at the same coverage level.
Method diversity and training image count change together, so this test does not isolate their effects. The test compares four alternatives per scene; its LOL-prior values differ from the five-method S14 evaluation.
Completed score conditions: 4500/4500; failed conditions: 0.
See README.md for reproduction commands and contrasts.csv for all paired intervals.
