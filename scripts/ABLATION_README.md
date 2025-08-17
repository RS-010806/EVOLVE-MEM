# EVOLVE-MEM Ablation Study (Standalone)

This document describes the standalone ablation framework that evaluates the contribution of EVOLVE-MEM components without modifying the core codebase.

## Goals
- Fair, fast comparisons across variants using a uniform sample (single story) by default
- Full-dataset reruns for top variants once ranked
- Zero changes to core files; all orchestration happens in `run_ablation.py`

## What varies
- Hierarchy usage: L0-only, L0+L1, L0+L2, Full (L0+L1+L2)
- Clustering schedule/shape: frequency ∈ {5,10,15}; fixed vs dynamic cluster counts (e.g., L1∈{3,5}, L2∈{1,3})
- Self-Improvement: on/off
- Answer patching: on/off

## Uniform sampling
`run_ablation.py` fixes a single `story_id` across all variants by default:
- `SAMPLE_MODE=True`
- `SAMPLE_STORY_INDEX=0`
This guarantees every variant runs on identical data.

## How to run
```bash
# 1) Fast ranking on a single uniform sample
python run_ablation.py

# 2) Full-dataset results for top variants
#    Open run_ablation.py, set SAMPLE_MODE=False and rerun selected variants
```

## Outputs
- Per-variant: `results/ablation_<variant>_<timestamp>.json`
- Summary table: `results/ablation_summary_<timestamp>.json`
- Log: `logs/ablation_<timestamp>.log`

## Metrics per variant
- Overall: Accuracy, F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT, Token Length
- By category: accuracy/F1
- Retrieval stats: Level 0/1/2/failed counts

## Reproducibility
- Uses the same data loader and evaluation module as the main pipeline
- Saves raw per-variant JSONs and a summary JSON for auditability

## Notes
- The ablation suite uses only public attributes of the system; no core edits required
- If needed, create additional variants by copying entries in `EXPERIMENTS` in `run_ablation.py`
