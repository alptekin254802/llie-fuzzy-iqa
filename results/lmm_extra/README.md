# Additional LMM evidence

VQA² and VITAL use released checkpoints and official image prompts without LLIE fitting. The saved results and paired intervals are the accepted NF4/FP32 + CPU-offload runs; Q-Align accuracy is fp16.

Model settings, precision checks, timing and recorded source availability: [evaluation protocol](#multimodal-evaluation-protocol).

`overlap_verified.json` groups all rated SQUARE-LOL scenes using normal-light source matches (231 shared and 59 non-overlapping); the classification was committed before subgroup recomputation. No images or checkpoint weights are included.

## Reproduce scores from images

Use separate Python 3.11 environments for `requirements-vqa2.txt` and `requirements-vital.txt`; their timm versions differ. For the recorded CUDA environment, PyTorch wheels use CUDA 11.8. Do not install the author training requirements over these pinned inference environments. Author code and weights are external dependencies and are not redistributed.

1. Obtain [VQA² source](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) at commit `9087c7952052088a6eb01bac4408bff903ab9e41` and [VITAL source](https://github.com/jzhws/VITAL-Series) at `729097bd45a799535c2e70297fbc96e81c317099`. VQA² uses Apache-2.0. VITAL has no root licence in the checked release; its InternVL files carry MIT headers. Use the author release directly under its applicable terms.
2. Download HF snapshots `q-future/VQA-Assistant-llava-qwen-enhanced` at `eff4328c0f3b4a6e9da98e5a608f3984ae2ff30d` and `JZHWS/VITAL-Base-8B` at `b8807b21c1d6442c0c5286038176523ab9943c40`, and `slowfast.pth` from `JZHWS/slowfast`. Source and checkpoint SHA-256 lists and the SlowFast hash are included here. The runner rejects different bytes before loading.
3. Copy `lmm_config.example.json` to `lmm_config.local.json`; set the local source/checkpoint locations and the three enhanced-image directories. Paths are relative to the configuration file. The ordered image manifest includes only dataset identities and RGB checksums, with no LIEQ/RLIE labels.
4. Run the commands below from the repository root, using the corresponding environment for each model. `bf16_cpu --pilot` reproduces the unquantized CPU reference; `nf4_fp32 --pilot` reproduces the 20-image comparison. Omit `--pilot` and supply the accepted pilot decision for all 5,440 images. `--limit 1 --pilot` is a short execution check, not a replacement for pilot acceptance.

```bash
python -B -m analysis.lmm_extra.run --model vqa2 --config lmm_config.local.json --output study_work/vqa2_reference --precision bf16_cpu --pilot
python -B -m analysis.lmm_extra.run --model vqa2 --config lmm_config.local.json --output study_work/vqa2_pilot --precision nf4_fp32 --pilot
python -B -m analysis.lmm_extra.verify --model vqa2 --reference study_work/vqa2_reference/scores.csv --candidate study_work/vqa2_pilot/scores.csv --output study_work/vqa2_decision.json
python -B -m analysis.lmm_extra.run --model vqa2 --config lmm_config.local.json --output study_work/vqa2_full --pilot-decision study_work/vqa2_decision.json
# Repeat with --model vital in its separate environment.
python -B -m analysis.lmm_extra.verify
```

Each row is flushed to CSV; rerunning the same command checks the existing prefix and resumes. NaN/inf stops inference. Full runs exclude nine warmup calls from their timing.

`analysis.lmm_extra.verify` reconstructs every archived score from its five logits, matches it to `joined.csv` and checks the recorded 20-image precision comparisons. The runtime metadata wrapper can be checked independently on CUDA with `python -B -m analysis.lmm_extra.test_nf4_compat --output study_work/nf4_check`. The public wrapper retains the measured computation while replacing private directories and experiment scheduling with explicit arguments.

## Human labels and reproduction

`joined.csv` retains model predictions and SQUARE-LOL MOS; LIEQ and RLIE MOS cells are intentionally empty. Obtain LIEQ (`MOS.mat`, `imgNames.mat`) and RLIE (`normalized_scores.csv`) from their authors. The released summary statistics do not require image downloads to inspect. SQUARE-LOL score attribution and permission are in [NOTICE](../../NOTICE).

From the repository root, recompute the comparisons with locally obtained labels:

```bash
python -B -m analysis.lmm_extra.analyze --lieq-root /path/to/LIEQ --rlie-root /path/to/RLIE --output study_work/lmm_extra
```

The command joins labels by dataset and exact image filename, checks for duplicate or missing labels, and reproduces all points, 2,000-draw scene-bootstrap intervals, paired differences and per-scene records against the released results. It does not run models or change their predictions. Local outputs stay under ignored `study_work/` (or an external directory); do not redistribute joined LIEQ/RLIE labels.

For a separate local joined input, including use with the Q-Align analysis, run:

```bash
python -B -m analysis.human_labels --lieq-root /path/to/LIEQ --rlie-root /path/to/RLIE --output study_work/lmm_extra/joined.csv
```

The roots may alternatively be set with `lieq` and `rlie` in `LLIE_STUDY_CONFIG`. Source hashes in the archived provenance identify the original unredacted records; the distribution entry identifies the released label-redacted file.

<a id="multimodal-evaluation-protocol"></a>

## VQA² and VITAL: multimodal evaluation protocol

This addition records the two accepted LMM runs. Released checkpoints and official image prompts were selected before LLIE outcomes; no fitting, fine-tuning, generation sampling or prompt search was performed. Q-Align accuracy remains the fp16 reference and its existing timing remains the separate 4-bit GPU measurement.

The preregistered v2 plan was recorded at 2026-10-23T00:32:50.858972+00:00 (SHA-256 `9c79d036cfb0ecfcd3e680e49dad09e7f51b01b6c4e47004b6adcb2dc68ab648`). Inputs use unchanged cv2 BGR2RGB and 256x256 INTER_AREA; SHA256 uint8 RGB compared to the earlier fp16 Q-Align run for every call; official native processor and prompts thereafter.

### Shared score calculation

One deterministic forward pass supplies the quality logits at position -3, token IDs [1550, 1661, 6624, 7852, 3347]. Float64 softmax and weights [1, 0.75, 0.5, 0.25, 0] produce the higher-is-better score. The source weights are a positive affine transformation of this published five-level equation; rank metrics and ties are unchanged.

### VQA²

Checkpoint `q-future/VQA-Assistant-llava-qwen-enhanced`, HF revision `eff4328c0f3b4a6e9da98e5a608f3984ae2ff30d`; official source commit `9087c7952052088a6eb01bac4408bff903ab9e41`. Image entry: `quality_scoring/llava/eval/model_score_image.py`.

Official prompt:

```json
{
  "system": "You are a helpful assistant.",
  "human": "<image>",
  "assistant_prefix": "The overall quality of the image is",
  "source_commit": "9087c7952052088a6eb01bac4408bff903ab9e41",
  "source": "quality_scoring/llava/eval/model_score_image.py:preprocess_qwen"
}
```

The released helper uses mismatched endpoint keys (high/low versus excellent/bad). The stored score follows the published five-token equation; the defective-key scalar is retained only as a diagnostic. Runtime casts use the requested dtype instead of a hard-coded half cast, with device-aware transfers; the forward algebra and official prompt are unchanged.

Device map and environment:

```json
{
  "device_map": {
    "model.image_newline": "0",
    "model.embed_tokens": "0",
    "model.layers.0": "0",
    "model.layers.1": "0",
    "model.layers.2": "0",
    "model.layers.3": "0",
    "model.layers.4": "cpu",
    "model.layers.5": "cpu",
    "model.layers.6": "cpu",
    "model.layers.7": "cpu",
    "model.layers.8": "cpu",
    "model.layers.9": "cpu",
    "model.layers.10": "cpu",
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": "cpu",
    "model.layers.15": "cpu",
    "model.layers.16": "cpu",
    "model.layers.17": "cpu",
    "model.layers.18": "cpu",
    "model.layers.19": "cpu",
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.24": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.norm": "cpu",
    "model.vision_tower": "cpu",
    "model.vision_resampler": "cpu",
    "model.mm_projector": "cpu",
    "model.slowfast": "cpu",
    "model.slowfast_projector": "cpu",
    "lm_head": "cpu"
  },
  "versions": {
    "torch": "2.7.1+cu118",
    "torchvision": "0.22.1+cu118",
    "transformers": "4.44.0",
    "accelerate": "0.34.2",
    "bitsandbytes": "0.49.2",
    "numpy": "2.4.4",
    "opencv-python": "4.13.0.92",
    "timm": "1.0.27"
  },
  "candidate_cpu_threads": 1,
  "reference_cpu_threads": 8
}
```

### VITAL

Checkpoint `JZHWS/VITAL-Base-8B`, HF revision `b8807b21c1d6442c0c5286038176523ab9943c40`; official source commit `729097bd45a799535c2e70297fbc96e81c317099`. Image entry: `VITAL-LMM/test/shell/eval/evaluate_image.sh -> internvl/eval/internvl_chat_finetune.py`.

Official prompt:

```json
{
  "system": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
  "human": "<image>\n",
  "assistant_prefix": "The quality of this image is",
  "template": "internvl2_5",
  "source_commit": "729097bd45a799535c2e70297fbc96e81c317099"
}
```

The actual image-evaluation shell route uses quality position -3; the unrelated rating.py last-position route is not used. Checkpoint keys are strictly matched, including the motion branch; redundant SlowFast pretrained initialization is omitted without omitting the checkpoint branch.

Device map and environment:

```json
{
  "device_map": {
    "vision_model": "0",
    "mlp1": "cpu",
    "motion_mlp": "cpu",
    "slowfast_model": "cpu",
    "language_model": "cpu"
  },
  "versions": {
    "torch": "2.7.1+cu118",
    "torchvision": "0.22.1+cu118",
    "transformers": "4.44.0",
    "accelerate": "0.34.2",
    "bitsandbytes": "0.49.2",
    "numpy": "2.4.4",
    "opencv-python": "4.13.0.92",
    "timm": "1.0.9"
  },
  "candidate_cpu_threads": 1,
  "reference_cpu_threads": 8
}
```

### Precision, diagnosis and pilot acceptance

Both released checkpoints store floating tensors in BF16. The CPU reference uses the unquantized bfloat16 checkpoint, as stated in the supplement. The CPU reference uses eager attention and the preregistered pilot order. The amended reference protocol uses eight CPU threads; candidate, diagnostic and full-run cost measurements retain one CPU thread. The original slow one-thread attempt is preserved and excluded from reference timing.

The sole candidate configuration uses NF4 weights, float32 computation and float32 nonquantized modules, with double quantization disabled. The GPU attempt exceeded physical GTX 1060 memory, invoking the prescribed automatic placement with a 5 GiB GPU budget and CPU remainder. CPU-mapped buffers follow their weights. The installed Transformers switch llm_int8_enable_fp32_cpu_offload enables this mixed mapping; no 8-bit quantization is used. Accelerate/bitsandbytes metadata transfer required an in-memory compatibility wrapper, verified with unchanged packed weights, metadata and forward outputs in a separate synthetic round-trip check. Full repair and recovery details remain in `results/lmm_extra/implementation_notes_v2.json`.

The fp16 diagnosis records the first observed nonfinite module boundary without changing any activation. Intentional attention-mask infinities are excluded. Both CPU BF16 diagnostic outputs are finite; both fp16 diagnoses first observe NaNs at the input of layer 27 attention output projection. A boundary observation does not isolate every internal operation.

| Model | First nonfinite input | Shape | NaNs |
|---|---|---|---:|
| VQA² | `model.layers.27.self_attn.o_proj` | [1, 784, 3584] | 148992 |
| VITAL | `language_model.model.layers.27.self_attn.o_proj` | [1, 312, 3584] | 16512 |

Pilot acceptance required finite scores on every selected image, SRCC at least 0.98 against the unquantized BF16 CPU reference, and positive Pearson and SRCC against frozen Q-Align. This rule was evaluated before human-score outcomes. A slow-reference rule and the dated CPU-reference amendment limited both accepted comparisons to the same first 20 pilot images; neither model was accepted through a later precision search.

| Model | Pilot N | CPU Pearson | CPU SRCC | Q-Align Pearson | Q-Align SRCC | Accepted |
|---|---:|---:|---:|---:|---:|---|
| VQA² | 20 | 0.977479 | 0.984962 | 0.762998 | 0.800000 | True |
| VITAL | 20 | 0.998868 | 0.993985 | 0.953367 | 0.959398 | True |

### Full evaluation, uncertainty and timing

Both models scored 5440 images, in SQUARE-LOL, RLIE, LIEQ order, with batch size one and seed 20261018. Every call checked the area-interpolated RGB hash against the fixed Q-Align input. Per-image records were appended and flushed, enabling verified resume without duplicating scores. Scene-mean SRCC uses average ranks and zero for a constant prediction, pair accuracy omits human ties and gives half credit to prediction ties, and random top-1 uses seed 20261021. Percentile 95% intervals use 2000 whole-scene draws with seed 20261018; paired draws keep both scorers. The added models have descriptive intervals and no hypothesis tests. `results.json`, `per_scene.csv` and `paired_differences.csv` preserve the summary results. The public `joined.csv` retains predictions and SQUARE-LOL scores; LIEQ and RLIE per-image labels are joined locally as described in [the reproduction guide](#human-labels-and-reproduction).

Timing uses the accepted NF4/FP32 configuration with CPU offloading. It excludes disk read, outer area resizing and dummy H2D transfer, and includes official preprocessing plus synchronized forward scoring. These are measured hybrid CPU/GPU costs forced by the 6 GB GPU, not estimates of an unrestricted GPU implementation.

| Model | Warmup N | Measured N | Median s | Mean s | P95 s |
|---|---:|---:|---:|---:|---:|
| VQA² | 9 | 5440 | 9.689107 | 9.857265 | 10.774728 |
| VITAL | 9 | 5440 | 5.672130 | 5.928172 | 6.596211 |

### Dated source availability

The latest recorded check is 23 October 2026. `availability_github_20261023.json` checked the official repositories at 2026-10-23 01:05:19 UTC; `availability_hf_extra_20261023.json` records the final related HF searches at 2026-10-23T02:37:45.573259+00:00. OmniVQA supplied a short README and license; the related HF repository had no model files. Refine-IQA listed testing code and weights as forthcoming while training code was present. These are dated observations, not claims about future availability. MINT-IQA expects generation prompts absent from the LLIE datasets.

### Source-overlap and independent alignment check

The normal-light classification gives 231 shared and 59 non-overlapping scenes; low00710 is shared despite its missing low-light source. VITAL scores were reconstructed from five raw logits and joined through independently recomputed RGB content hashes; all 2900 image identities and 290 scene correlations agreed, to maximum absolute errors 2.22e-16 and 1.11e-16. Its mean within-scene SRCC is 0.344460 on shared scenes and -0.105699 on verified new scenes. No per-scene source-collection labels were available. The negative new-scene result is retained.
