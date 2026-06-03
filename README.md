# PreSimNet

PreSimNet is a physics-encoded mixture-of-experts model for heterogeneous car-following trajectory prediction. The active implementation uses four interaction-type experts and supports Top-2 expert aggregation by selecting the two largest predicted routing probabilities and renormalizing them before dynamic prediction.

This repository is cleaned for code release: full datasets, checkpoints, training logs, and large visualization dumps are excluded. A small `.npy` sample is included only to document the data schema and support smoke tests.

## Repository Layout

- `train_moe_top2.py`: recommended Top-2 training entry point.
- `model/model_MoE_gru_new.py`: active PreSimNet model. `predictor.route_probabilities` supports `soft_all`, `hard_top1`, and `top2`.
- `analyze_routing_topk.py`: routing analysis and evaluation-style RMSE for hard Top-1, Top-2, and related subsets.
- `evaluate_position_rmse.py`: direct position prediction RMSE comparison.
- `speed_stratified_eval.py` and `loss_sensitivity_top2.py`: supplementary analysis scripts.
- `train_baseline.py`, `evaluate_baseline.py`, and `model/*_baseline.py`: baseline model training and evaluation code.
- `loader2.py`: dataset loader and column schema.
- `data/sample/test_data_sample.npy`: 128-window example sample from the test set.
- `docs/reports/`: project reports used as the source of the reported metrics.
- `docs/results/top2_trained/`: CSV tables from the latest Top-2 full-test analysis.
- `docs/figures/top2_trained/`: selected routing-analysis figures.

## Data Format

The full data files are NumPy arrays with shape `(N, 40, 13)`. Each sample contains 20 historical steps and 20 prediction steps. Column meanings are:

| Index | Meaning |
|---:|---|
| 0 | dataset id |
| 1 | trajectory id |
| 2 | timestamp |
| 3 | car-following interaction type |
| 4 | lead vehicle velocity |
| 5 | lead vehicle acceleration |
| 6 | following vehicle velocity |
| 7 | following vehicle acceleration |
| 8 | gap |
| 9 | headway |
| 10 | relative velocity |
| 11 | following vehicle position |
| 12 | lead vehicle position |

Interaction type ids follow the project reports: `0=AV-HV`, `1=AV-AV`, `2=HV-HV`, `3=HV-AV`.

## Setup

Install a PyTorch build matching your CUDA environment, then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

For this remote machine, GPU runs were validated in WSL with Python at `/home/codex/venvs/unipe/bin/python`.

## Smoke Test

The included sample is not large enough for real training, but it is useful for checking imports, the loader, and a short forward/training loop:

```powershell
python train_moe_top2.py --train-data data/sample/test_data_sample.npy --epochs 1 --batch-size 16 --max-batches 2 --checkpoint-dir runs/sample/checkpoints --result-dir runs/sample/results
```

## Training

Run Top-2 training on the full data:

```powershell
python train_moe_top2.py --train-data ../data/train_data.npy --epochs 21 --batch-size 512 --num-workers 4 --gamma 0.9 --data-loading mmap
```

The default output directories follow the project convention:

- checkpoints: `checkponint/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2/`
- logs: `result/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2/`

## Evaluation

Latest Top-2 routing analysis:

```powershell
python analyze_routing_topk.py --checkpoint checkponint/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2/epoch21_e.tar --data ../data/test_data.npy --out-dir fig/vis/routing_top2_trained_full_test --batch-size 4096 --device cuda:0
```

Direct position RMSE comparison:

```powershell
python evaluate_position_rmse.py --data ../data/test_data.npy --batch-size 4096 --device cuda:0
```

## Latest Top-2 Metrics

The following metrics are taken from `docs/reports/top2_trained_report_zh.md`, which is the latest project report for the Top-2 full-test setting.

Full test RMSE:

| Routing | Gap 0.5s | Vel 0.5s | Gap 1.0s | Vel 1.0s | Gap 1.5s | Vel 1.5s | Gap 2.0s | Vel 2.0s | Gap Avg | Vel Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-2 | 0.230759 | 0.266919 | 0.364927 | 0.307181 | 0.483280 | 0.361042 | 0.621276 | 0.452345 | 0.366770 | 0.310733 |

Routing identification statistics:

| Type | Samples | Hard Top-1 errors | Hard Top-1 error rate | Top-2 misses | Top-2 miss rate | Top-2 rescues | Rescue / hard error |
|---|---:|---:|---:|---:|---:|---:|---:|
| AV-AV | 404,724 | 1,974 | 0.488% | 14 | 0.003% | 1,960 | 99.29% |
| AV-HV | 552,061 | 8,059 | 1.460% | 93 | 0.017% | 7,966 | 98.85% |
| HV-AV | 543,312 | 3,232 | 0.595% | 99 | 0.018% | 3,133 | 96.94% |
| HV-HV | 746,842 | 7,554 | 1.011% | 186 | 0.025% | 7,368 | 97.54% |
| ALL | 2,246,939 | 20,819 | 0.927% | 392 | 0.017% | 20,427 | 98.12% |

Hard Top-1 error subset RMSE:

| Routing | Samples | Gap Avg | Vel Avg | Gap 2.0s | Vel 2.0s |
|---|---:|---:|---:|---:|---:|
| Soft-all | 20,819 | 0.368523 | 0.217419 | 0.624424 | 0.410484 |
| Hard Top-1 | 20,819 | 0.371285 | 0.221295 | 0.632007 | 0.416181 |
| Top-2 | 20,819 | 0.368588 | 0.217547 | 0.624638 | 0.410693 |

Direct position prediction RMSE:

| Model | Samples | Pos 0.5s | Pos 1.0s | Pos 1.5s | Pos 2.0s | Pos Avg |
|---|---:|---:|---:|---:|---:|---:|
| Original soft-all weights | 2,246,939 | 0.138933 | 0.243327 | 0.364692 | 0.510625 | 0.260479 |
| Top-2 retrained weights | 2,246,939 | 0.142280 | 0.249773 | 0.370602 | 0.517349 | 0.265333 |

## Notes

- Full data and checkpoints are intentionally not versioned. Place full data under `../data/` or pass explicit paths through CLI arguments.
- The historical `evaluate_new.py` points to an older ablation model. For the latest Top-2 routing metrics, use `analyze_routing_topk.py` with `model/model_MoE_gru_new.py`.
- `checkponint` is the original project spelling and is preserved for compatibility with existing scripts.
