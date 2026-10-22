# GATE reviewer attribution controls (CLIP / ITSELF)

This patch adds four **separate** controls. It does **not** change the default full-GATE path, original retrieval loss, checkpoint architecture, score interpolation grid, or the evaluation protocol. The experiments test (i) gallery context vs added trainable capacity, (ii) reliance on correct candidate–evidence association, (iii) similarity-weighted gallery feedback, and (iv) direct uniform Rocchio PRF without fusion.

| Control | CLI | Needs training? | What it tests |
|---|---|---|---|
| **GATE-Full** (unchanged) | `--gate_variant full` (default) | Existing checkpoint | Reference |
| **V1 QueryOnly** | `--gate_variant query_only` during training | **Yes** | Same weights/parameter shapes, supervision, and QCRS path but no gallery in the query correction |
| **V2 ShuffledEvidence** | `--evidence_control shuffle --evidence_seed 11` in `test.py` | **No** | Correct correspondence of selected candidates to evidence |
| **V3 Similarity-Weighted QE** | `--eval_baseline similarity_qe` in `test.py` | **No** | Simple rank/similarity-based gallery feedback |
| **V4 Uniform Rocchio PRF** | `--eval_baseline uniform_prf` in `test.py` | **No** | Equal-weight Top-K gallery evidence without a score-fusion sweep |

## Requirements / source of truth

Run from **`T-TBPS/itself`**. Training accepts **CLI options** and saves a resolved `configs.yaml` with the checkpoint; `train.py` does not load `itself/configs/{clip,itself}_base.yaml` automatically. **Start from the exact args/checkpoint of the Full GATE run being compared**, not from a different experimental template. The provided `run_clip.sh`/`run_itself.sh` are historical examples and use at least one invalid `--enrichment_space retrieval` value; do **not** use them unchanged. Accepted values are `global` and `grab`. Host checkpoints and datasets must be supplied separately.

Typical environment:

```bash
cd /path/to/T-TBPS/itself
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_gate_controls.py" -v
```

### Reference Full (no changes)

Use the existing GATE checkpoint and its own resolved config:

```bash
python test.py --run_dir /runs/clip_full_gate \
  --source_domain CUHK-PEDES --target_domain CUHK-PEDES \
  --target_enrichment --save_json --output_eval_dir /runs/eval/full
```

### V1: QueryOnly — train one NEW checkpoint per source setting

QueryOnly replaces the gallery evidence tokens with **query-derived tokens repeated across the original M×S positions (in GRAB space, reuse the original `proto_to_grab` projection of the global text query)**. The same QCRS weights/parameter names/shapes, learned rank embeddings, FiLM, residual fusion, frozen host and supervised source-gallery retrieval objective are retained. Test-time query correction does not read any gallery data. Training loss still uses source-gallery image embeddings/identity labels, as in conventional retrieval supervision.

CLIP example (edit paths and **match the actual full-run hyperparameters**):

```bash
python train.py --dataset_name CUHK-PEDES --root_dir /data/tbps \
  --finetune_clip /checkpoints/clip_host_best.pth \
  --output_dir /runs/gate_queryonly_clip \
  --target_enrichment --gate_variant query_only \
  --freeze_host --no_use_host_loss --only_global \
  --pnp_text_only --use_freeze_indices --recompute_interval -1 \
  --enrichment_space global --topm_rank_space host_global \
  --extractor_mode global --top_m 64 --mixer_depth 1 \
  --mixer_hidden_rank 128 --mixer_hidden_readout 256 \
  --residual_gate static --enrich_gamma 0.1 --lambda_ret 0.5 \
  --num_epoch 30 --lr_total_epochs 60
```

ITSELF example (uses GRAB + hybrid initial rank, adapt to your actual Full ITSELF run):

```bash
python train.py --dataset_name CUHK-PEDES --root_dir /data/tbps \
  --finetune_clip /checkpoints/itself_host_best.pth \
  --output_dir /runs/gate_queryonly_itself \
  --target_enrichment --gate_variant query_only \
  --freeze_host --no_use_host_loss --use_freeze_indices --recompute_interval -1 \
  --return_all --topk_type custom --modify_k \
  --enrichment_space grab --topm_rank_space hybrid_global_grab \
  --topm_rank_lambda 0.4 --extractor_mode global --top_m 64 \
  --mixer_depth 1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 \
  --residual_gate static --enrich_gamma 0.1 --lambda_ret 0.5 \
  --num_epoch 30 --lr_total_epochs 60
```

Inference reads `gate_variant: query_only` automatically from the saved run config:

```bash
python test.py --run_dir /runs/gate_queryonly_clip/CUHK-PEDES/<RUN_DIR> \
  --source_domain CUHK-PEDES --target_domain CUHK-PEDES \
  --target_enrichment --save_json --output_eval_dir /runs/eval/queryonly
```

`train.py` nests `output_dir / dataset_name / timestamp_name_loss`; locate the generated run directory containing `configs.yaml` and `best.pth`. When running ITSELF, substitute its run directory, base checkpoint and corresponding args. V1 is the **only** newly trainable variant.

### V2: ShuffledEvidence — reuse the exact GATE-Full checkpoint

```bash
python test.py --run_dir /runs/clip_full_gate \
  --source_domain CUHK-PEDES --target_domain CUHK-PEDES \
  --target_enrichment --evidence_control shuffle --evidence_seed 11 \
  --save_json --output_eval_dir /runs/eval/shuffle11
```

Repeat with seeds 11, 23, 37. One deterministic permutation is created **per evaluation gallery** and shared across queries. It permutes `evidence_bank` and any candidate-associated auxiliary evidence, but **never changes** retrieval features, PIDs, Top-M indices, or final-gallery ranking candidates. The trained GATE weights remain fixed. It is an inference intervention and may cause distribution shift, so do not treat a performance drop alone as proof of causal benefit.

### V3: Similarity-Weighted QE — frozen host only, no GATE checkpoint/training

```bash
python test.py --run_dir /runs/clip_host \
  --source_domain CUHK-PEDES --target_domain CUHK-PEDES \
  --no_target_enrichment --eval_baseline similarity_qe \
  --qe_top_m 64 --qe_alpha 0.5 --qe_temperature 0.05 \
  --enrichment_space global --save_json --output_eval_dir /runs/eval/qe
```

For ITSELF, use its frozen host config/checkpoint, with `--enrichment_space grab` **if** the Full GATE refinement was in GRAB space; the current host `topm_rank_space` is read from config (e.g. `hybrid_global_grab`). QE is `normalize(q + alpha * sum(softmax(top_m_scores / temperature) * image_embedding))`. Rank selection follows host `topm_rank_space`; evidence vectors use the same embedding space as the refined query. Scores still cover the **entire gallery**. Results are labeled `global+qe(...)` or `grab+qe(...)` rather than `+proto(...)`. The evaluator applies the same final score-interpolation grid to the QE query as to GATE.

`--qe_alpha`, `--qe_temperature`, `--qe_top_m` are explicit hyperparameters; report their values and select them under the same experimental policy as Full GATE. `test.py --target_enrichment` and `--eval_baseline similarity_qe` cannot be combined.

### V4: Uniform Rocchio PRF — independent frozen-host control

V4 uses only **original frozen BASE host features**, not GATE weights or a trained query mixer. For normalized query `q` and normalized gallery embeddings `g`, choose Top-K indices using the configured initial `topm_rank_space`, compute `c_K = mean(g[TopK])` (uniform weighting), and rank the **entire gallery** with `normalize(q + beta * c_K) @ g.T`. Defaults are `--prf_top_k 64` and `--prf_beta 0.5`. No labels, softmax, QE temperature, learned parameters, score scaling, or final fusion lambda are used.

**V3 versus V4:** V3 weights candidates with softmax similarity and evaluates QE inside the existing GATE-like lambda-fusion sweep. V4 gives every selected candidate exactly equal weight and reports the **direct, unfused PRF similarity ranking**, even if its R@1 is worse than BASE. In V4's output, `best_task` always names `global+uniform_prf` or `grab+uniform_prf`; `best_R1` is the PRF R@1 rather than the best of BASE and PRF. No `eval/ablation_best_lambda` exists.

**CLIP (original frozen BASE checkpoint only; NOT a GATE checkpoint):**

```bash
python test.py \
  --run_dir /path/to/clip_base_run \
  --checkpoint /path/to/original_frozen_clip.pth \
  --root_dir /path/to/tbps-benchmark/benchmark \
  --source_domain ICFG-PEDES \
  --target_domain ICFG-PEDES \
  --no_target_enrichment \
  --only_global \
  --eval_baseline uniform_prf \
  --prf_top_k 64 \
  --prf_beta 0.5 \
  --enrichment_space global \
  --save_json --output_eval_dir /runs/eval/uniform_prf_icfg
```

For ITSELF, pass the **original frozen ITSELF BASE checkpoint**, `--no_only_global --enrichment_space grab` for a GRAB experiment, and retain the Full GATE run's `topm_rank_space`/`topm_rank_lambda` policy in its config (or select `global` for global-space PRF). The rank policy can be `host_global`, `retrieval`, or `hybrid_global_grab`; only Top-K **selection** may use another/hybrid space. PRF aggregation and final scoring always use the same chosen `enrichment_space` embeddings. A GRAB or hybrid selection requires GRAB features; `--only_global` cannot be combined with them.

V4 produces exactly two text-to-image rows: frozen `global` (or `grab`) and `global+uniform_prf` (or `grab+uniform_prf`). The JSON includes both rows' `R1`, `R5`, `R10`, `mAP`, `mINP`, plus direct `eval/prf_R1`, `eval/prf_R5`, `eval/prf_R10`, `eval/prf_mAP`, `eval/prf_mINP`, `eval/prf_top_k`, and `eval/prf_beta`. `--save_json`, `--results_file`, and `--output_eval_dir` remain supported. Select `top_k` and `beta` without tuning on the benchmark test split.

## Checkpoint and reporting rules

- Keep Full GATE checkpoints untouched; QueryOnly's `gate_variant` is saved in its own `configs.yaml`.
- Record `source_domain`, `target_domain`, `base_checkpoint`, `gate_variant`, `top_m`, `extractor_mode`, `enrichment_space`, `topm_rank_space`, final lambda, and seeds.
- `test.py --save_json` writes an `inference_control` block and metrics `eval/control_*`, plus QE hyperparameters where relevant.
- Report paired **R@1 and mAP**, and multiple V2 seeds. QueryOnly should be trained with the Full GATE budget, not compared to a differently tuned Full run.
- **No claim that V2 provides a capacity-matched causal test**: that is the role of V1. V3 tests generic gallery feedback vs a learned GATE mechanism, not setting-level novelty.
- To avoid changing the current manuscript's protocol implicitly, this patch does not modify checkpoint selection or the existing evaluator's final lambda sweep for Main/V1/V2/V3; only V4 deliberately bypasses that sweep.

## Sanity checks

```bash
cd /path/to/T-TBPS/itself
python -m unittest discover -s tests -p "test_gate_controls.py" -v
python -m unittest discover -s tests -v
```

V1: `model.target_enricher` has exactly the same parameter-key/shape inventory as Full, and `enrich_only(q, ..., None)` must be invariant to gallery changes. V2: identical seed gives identical permutation; gallery retrieval embeddings and PIDs remain unmodified. V3: QE with `--qe_alpha 0` must recover base similarity scores for the matching feature space. V4: `--prf_beta 0` recovers the frozen baseline; Top-K aggregation is uniform; only full-gallery PRF scores are reported, even if the baseline wins.
