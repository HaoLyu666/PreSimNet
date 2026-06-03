from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch as t
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import model.model_MoE_gru_new as model
from config import args as base_args
from config import learning_rate
from train_moe_top2 import (
    HighSimMemmapDataset,
    count_parameters,
    initialize_expert_biases,
    param_distribution_loss,
)


TYPE_LABELS = {
    0: "AV-HV",
    1: "AV-AV",
    2: "HV-HV",
    3: "HV-AV",
}

HORIZONS = {
    "0.5s": 4,
    "1.0s": 9,
    "1.5s": 14,
    "2.0s": 19,
}

ANALYSIS_DIR = Path("fig/vis/loss_sensitivity_top2")
CHECKPOINT_ROOT = Path("checkponint/loss_sensitivity_top2")
RESULT_ROOT = Path("result/loss_sensitivity_top2")
DEFAULT_TOP2_CHECKPOINT_DIR = Path(
    "checkponint/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2"
)
DEFAULT_TOP2_RESULT_DIR = Path(
    "result/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2"
)
FAST_REQUIRED_ONLY_SENTINEL = ANALYSIS_DIR / "FAST_REQUIRED_ONLY"
REQUIRED_FULL_VARIANTS = {
    "default_raw",
    "sim_half_raw",
    "sim_double_raw",
    "vel_emphasis_raw",
    "gap_emphasis_raw",
    "strong_param_raw",
    "default_normalized",
}

RUN_CONFIG_FIELDS = [
    "variant",
    "alpha_pos",
    "alpha_vel",
    "alpha_gap",
    "alpha_type",
    "lambda_param",
    "loss_mode",
    "routing_mode",
    "seed",
    "python_random_seed",
    "run_scope",
    "status",
    "selection_rule",
    "checkpoint_dir",
    "checkpoint_path",
    "result_dir",
    "command",
    "notes",
]

TRAIN_LOG_FIELDS = [
    "variant",
    "epoch",
    "run_scope",
    "loss_mode",
    "routing_mode",
    "alpha_pos",
    "alpha_vel",
    "alpha_gap",
    "alpha_type",
    "lambda_param",
    "train_total_loss",
    "train_pos_loss",
    "train_vel_loss",
    "train_gap_loss",
    "train_type_loss",
    "train_param_loss",
    "validation_selection_metric",
    "val_pos_rmse_0.5s",
    "val_pos_rmse_1.0s",
    "val_pos_rmse_1.5s",
    "val_pos_rmse_2.0s",
    "val_pos_rmse_avg_4h",
    "val_vel_rmse_0.5s",
    "val_vel_rmse_1.0s",
    "val_vel_rmse_1.5s",
    "val_vel_rmse_2.0s",
    "val_vel_rmse_avg_4h",
    "val_gap_rmse_0.5s",
    "val_gap_rmse_1.0s",
    "val_gap_rmse_1.5s",
    "val_gap_rmse_2.0s",
    "val_gap_rmse_avg_4h",
    "val_type_acc_avg",
    "checkpoint_path",
    "is_best_checkpoint",
    "skipped_nan_batches",
]

SUMMARY_FIELDS = [
    "variant",
    "routing_mode",
    "loss_mode",
    "alpha_pos",
    "alpha_vel",
    "alpha_gap",
    "alpha_type",
    "lambda_param",
    "run_scope",
    "selection_rule",
    "selected_epoch",
    "selection_metric",
    "pos_rmse_0.5s",
    "pos_rmse_1.0s",
    "pos_rmse_1.5s",
    "pos_rmse_2.0s",
    "pos_rmse_avg_4h",
    "vel_rmse_0.5s",
    "vel_rmse_1.0s",
    "vel_rmse_1.5s",
    "vel_rmse_2.0s",
    "vel_rmse_avg_4h",
    "gap_rmse_0.5s",
    "gap_rmse_1.0s",
    "gap_rmse_1.5s",
    "gap_rmse_2.0s",
    "gap_rmse_avg_4h",
    "type_acc_AV_HV",
    "type_acc_AV_AV",
    "type_acc_HV_HV",
    "type_acc_HV_AV",
    "type_acc_avg",
    "param_reg_loss",
    "param_violation_rate",
    "param_violation_abs_mean",
    "negative_gap_rate",
    "acceleration_bound_violation_rate",
    "mean_abs_acceleration",
    "mean_abs_jerk",
    "sample_count",
    "checkpoint_path",
]

VARIANTS = [
    {
        "variant": "default_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "reuse_default": True,
    },
    {
        "variant": "sim_half_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 0.5,
        "alpha_gap": 0.5,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "sim_double_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 2.0,
        "alpha_gap": 2.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "vel_emphasis_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 2.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "gap_emphasis_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 2.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "weak_param_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.001,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "strong_param_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.1,
        "loss_mode": "raw",
        "reuse_default": False,
    },
    {
        "variant": "default_normalized",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "normalized",
        "reuse_default": False,
    },
]


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def release_variant_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def acquire_variant_lock(variant: str) -> Path | None:
    lock_dir = ANALYSIS_DIR / "variant_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{variant}.lock"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "variant": variant,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            atexit.register(release_variant_lock, lock_path)
            return lock_path
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(existing.get("pid", -1))
            except (OSError, ValueError, json.JSONDecodeError):
                pid = -1
            if pid > 0 and is_pid_alive(pid):
                print(
                    f"Variant {variant} is already running under pid {pid}; "
                    "skipping duplicate launch."
                )
                return None
            release_variant_lock(lock_path)
    return None


def set_seeds(torch_seed: int, python_seed: int) -> None:
    random.seed(python_seed)
    np.random.seed(torch_seed)
    t.manual_seed(torch_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(torch_seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def parse_epoch_from_checkpoint(path: Path) -> int | str:
    name = path.name
    if name == "best_e.tar":
        return "best"
    if name.startswith("epoch") and name.endswith("_e.tar"):
        try:
            return int(name[5:-6])
        except ValueError:
            return ""
    return ""


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def upsert_row(path: Path, key: tuple[str, ...], row: dict, fieldnames: list[str]) -> None:
    rows = read_csv_rows(path)
    key_values = tuple(str(row.get(k, "")) for k in key)
    updated = False
    for idx, existing in enumerate(rows):
        if tuple(str(existing.get(k, "")) for k in key) == key_values:
            rows[idx] = {**existing, **row}
            updated = True
            break
    if not updated:
        rows.append(row)
    write_csv_rows(path, rows, fieldnames)


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fmt(value)
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def base_run_args(cli: argparse.Namespace, train_flag: bool) -> dict:
    device = t.device(cli.device)
    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = train_flag
    run_args["gamma"] = cli.gamma
    run_args["epoch"] = cli.epochs
    run_args["last_epoch"] = cli.last_epoch
    run_args["batch_size"] = cli.batch_size
    run_args["num_worker"] = cli.num_workers
    run_args["aggregation_mode"] = "top2"
    run_args["route_top_k"] = 2
    return run_args


def variant_checkpoint_dir(variant: str) -> Path:
    return CHECKPOINT_ROOT / variant


def variant_result_dir(variant: str) -> Path:
    return RESULT_ROOT / variant


def save_implementation_check() -> None:
    ensure_dirs()
    text = """# Loss Sensitivity Top-2 Implementation Check

This note records implementation facts for the loss-sensitivity experiment package only. It is not reviewer response text and it does not edit the manuscript.

## Training Losses

- Current legacy `train_new.py` imports `model.model_MoE_gru_new_ablation` and computes `total_loss = direct_pos_loss + dynamic_gap_loss + dynamic_vel_loss + cf_type_loss + 0.01 * param_dist_loss`.
- Current Top-2 retraining uses `train_moe_top2.py`, which imports `model.model_MoE_gru_new` and computes the same five raw loss terms before weighting.
- `direct_pos_loss`: MSE between `direct_pred.squeeze(-1)` and future following-vehicle position `fut[:, :, 2]`.
- `dynamic_gap_loss`: MSE between closed-loop dynamic gap prediction `dynamic_pred[:, :, 0]` and future gap `fut[:, :, 0]`.
- `dynamic_vel_loss`: MSE between closed-loop dynamic velocity prediction `dynamic_pred[:, :, 1]` and future following-vehicle velocity `fut[:, :, 1]`.
- `cf_type_loss`: `CrossEntropyLoss(cf_type_pred, cf_type)`, where `cf_type` is the one-hot interaction label.
- `param_dist_loss`: physical-bound penalty over two ACC experts and two IDM experts. For each parameter, ReLU lower/upper bound violation is averaged over the batch; desired-speed-like parameters are divided by 10. Expert penalties are weighted by routing probabilities.

## Model File And Routing

- Official sensitivity scripts in this package use `model/model_MoE_gru_new.py`.
- `train_moe_top2.py` sets `aggregation_mode = "top2"` and `route_top_k = 2`.
- `model/model_MoE_gru_new.py` has `predictor.route_probabilities`; when `aggregation_mode` is unset it defaults to `soft_all`, and when set to `top2`/`topk` it selects the two largest predicted probabilities and renormalizes over the selected experts.
- `ParameterPredictionHead.forward` computes `cf_probs = softmax(cf_type_pred)`. These are model-predicted probabilities.

## Top-2 Closed-Loop Simulation Check

- Normal Top-2 inference uses predicted `cf_probs`, not ground-truth type labels.
- `predictor.veh_dynamic` computes four candidate accelerations: two ACC accelerations from ACC parameters and two IDM accelerations from IDM parameters.
- The Top-2 routing weights are applied to the candidate acceleration tensor only: `a_t = bmm(route_probs.unsqueeze(1), all_accelerations).squeeze(1)`.
- ACC and IDM parameter vectors are not mixed before model evaluation; only final candidate accelerations are aggregated.

## Experiment Package Behavior

- `default_raw` reuses the ongoing default Top-2 training output and does not retrain it.
- New sensitivity training outputs are isolated under `checkponint/loss_sensitivity_top2/` and `result/loss_sensitivity_top2/`.
- Official runs use the same split paths: `../data/train_data.npy`, `../data/valid_data.npy`, and `../data/test_data.npy`.
- `avg_4h` is defined as the mean of RMSE at 0.5s, 1.0s, 1.5s, and 2.0s.
- Test data are used only after validation selection.
"""
    (ANALYSIS_DIR / "implementation_check.md").write_text(text, encoding="utf-8")


def compute_loss_scale_statistics(train_data: str, output: Path) -> dict[str, float]:
    ensure_dirs()
    data = np.load(train_data, mmap_mode="r")
    n = len(data)
    sums = {
        "position": 0.0,
        "velocity": 0.0,
        "gap": 0.0,
    }
    sums_sq = {
        "position": 0.0,
        "velocity": 0.0,
        "gap": 0.0,
    }
    mins = {
        "position": float("inf"),
        "velocity": float("inf"),
        "gap": float("inf"),
    }
    maxs = {
        "position": float("-inf"),
        "velocity": float("-inf"),
        "gap": float("-inf"),
    }
    count = 0
    chunk_size = 8192
    for start in tqdm(range(0, n, chunk_size), desc="loss scales"):
        end = min(start + chunk_size, n)
        window = data[start:end, 20:, :]
        values = {
            "position": window[:, :, 11].astype(np.float64, copy=False),
            "velocity": window[:, :, 6].astype(np.float64, copy=False),
            "gap": window[:, :, 8].astype(np.float64, copy=False),
        }
        batch_count = values["position"].size
        count += batch_count
        for key, arr in values.items():
            sums[key] += float(arr.sum(dtype=np.float64))
            sums_sq[key] += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))
            mins[key] = min(mins[key], float(arr.min()))
            maxs[key] = max(maxs[key], float(arr.max()))

    rows = []
    scales = {}
    mapping = {
        "position": "sigma_pos",
        "velocity": "sigma_vel",
        "gap": "sigma_gap",
    }
    for key in ["position", "velocity", "gap"]:
        mean = sums[key] / count
        variance = max(sums_sq[key] / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        sigma = std if std > 1e-8 else 1.0
        scales[mapping[key]] = sigma
        rows.append(
            {
                "quantity": key,
                "sigma_name": mapping[key],
                "sigma": sigma,
                "mean": mean,
                "std": std,
                "min": mins[key],
                "max": maxs[key],
                "sample_count": count,
                "source_split": "train",
                "source_path": train_data,
            }
        )

    fields = [
        "quantity",
        "sigma_name",
        "sigma",
        "mean",
        "std",
        "min",
        "max",
        "sample_count",
        "source_split",
        "source_path",
    ]
    write_csv_rows(output, rows, fields)
    return scales


def load_loss_scales(path: Path, train_data: str) -> dict[str, float]:
    if not path.exists():
        return compute_loss_scale_statistics(train_data, path)
    scales = {}
    for row in read_csv_rows(path):
        scales[row["sigma_name"]] = float(row["sigma"])
    missing = {"sigma_pos", "sigma_vel", "sigma_gap"} - set(scales)
    if missing:
        return compute_loss_scale_statistics(train_data, path)
    return scales


class RMSEAccumulator:
    def __init__(self, out_length: int, device: t.device):
        self.sse = t.zeros(out_length, dtype=t.float64, device=device)
        self.counts = t.zeros(out_length, dtype=t.float64, device=device)

    def update(self, pred: t.Tensor, gt: t.Tensor) -> None:
        err = (pred.double() - gt.double()).pow(2)
        self.sse += err.sum(dim=0)
        self.counts += pred.shape[0]

    def result(self) -> np.ndarray:
        out = t.sqrt(self.sse / self.counts.clamp_min(1.0)).detach().cpu().numpy()
        out[self.counts.detach().cpu().numpy() == 0] = np.nan
        return out


def make_batch_iterator(
    data_path: str,
    batch_size: int,
    in_length: int,
    out_length: int,
    max_batches: int | None,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = np.load(data_path, mmap_mode="r")
    total = len(data)
    usable = total if max_batches is None else min(total, max_batches * batch_size)
    for start in range(0, usable, batch_size):
        end = min(start + batch_size, usable)
        window = data[start:end]
        hist_acc_diff = window[:, :, 5] - window[:, :, 7]
        hist = np.concatenate(
            [window[:, :in_length, 4:], hist_acc_diff[:, :in_length, None]],
            axis=2,
        ).astype(np.float32, copy=False)
        nextv = window[:, in_length - 1 :, 4].astype(np.float32, copy=False)
        fut = window[:, in_length:, [8, 6, 11, 12]].astype(np.float32, copy=False)
        target = window[:, 0, 3].astype(np.int64, copy=False)
        yield hist, nextv, fut, target


def make_train_batch_iterator(
    data_path: str,
    batch_size: int,
    in_length: int,
    out_length: int,
    epoch: int,
    seed: int,
    max_batches: int | None,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = np.load(data_path, mmap_mode="r")
    usable = (len(data) // batch_size) * batch_size
    starts = np.arange(0, usable, batch_size, dtype=np.int64)
    rng = np.random.default_rng(seed + epoch)
    rng.shuffle(starts)
    if max_batches is not None:
        starts = starts[:max_batches]

    for start in starts:
        end = int(start) + batch_size
        window = data[int(start) : end]
        hist_acc_diff = window[:, :, 5] - window[:, :, 7]
        hist = np.concatenate(
            [window[:, :in_length, 4:], hist_acc_diff[:, :in_length, None]],
            axis=2,
        ).astype(np.float32, copy=False)
        nextv = window[:, in_length - 1 :, 4].astype(np.float32, copy=False)
        fut = window[:, in_length:, [8, 6, 11, 12]].astype(np.float32, copy=False)
        cf_type_idx = window[:, 0, 3].astype(np.int64, copy=False)
        cf_type = np.zeros((window.shape[0], 4), dtype=np.float32)
        cf_type[np.arange(window.shape[0]), cf_type_idx] = 1.0
        yield hist, nextv, fut, cf_type


def compute_candidate_accelerations(
    simulator: model.predictor,
    params: dict,
    veh_state: t.Tensor,
    next_v_t: t.Tensor,
) -> t.Tensor:
    acc_params = params["acc_params"]
    idm_params = params["idm_params"]
    delta_v = next_v_t - veh_state[:, 1:2]
    a0 = simulator.acc_model(acc_params[0].squeeze(1), veh_state, next_v_t, delta_v)
    a1 = simulator.acc_model(acc_params[1].squeeze(1), veh_state, next_v_t, delta_v)
    a2 = simulator.idm_model(idm_params[0].squeeze(1), veh_state, delta_v)
    a3 = simulator.idm_model(idm_params[1].squeeze(1), veh_state, delta_v)
    return t.stack([a0, a1, a2, a3], dim=1)


def param_violation_diagnostics(acc_params, idm_params) -> tuple[t.Tensor, t.Tensor]:
    ranges = {
        "acc": [
            (0.01, 2.0),
            (0.01, 2.0),
            (5.0, 40.0),
            (0.5, 10.0),
            (0.0, 5.0),
            (0.01, 0.5),
        ],
        "idm": [
            (0.5, 10.0),
            (0.2, 5.0),
            (0.2, 4.0),
            (0.2, 5.0),
            (5.0, 50.0),
            (2.0, 6.0),
        ],
    }
    violations = []
    abs_values = []
    for expert_idx in range(2):
        params = acc_params[expert_idx].squeeze(1)
        for param_idx, (low, high) in enumerate(ranges["acc"]):
            value = params[:, param_idx]
            below = t.relu(low - value)
            above = t.relu(value - high)
            violation = below + above
            violations.append(violation > 0)
            abs_values.append(violation)
    for expert_idx in range(2):
        params = idm_params[expert_idx].squeeze(1)
        for param_idx, (low, high) in enumerate(ranges["idm"]):
            value = params[:, param_idx]
            below = t.relu(low - value)
            above = t.relu(value - high)
            violation = below + above
            violations.append(violation > 0)
            abs_values.append(violation)
    violation_tensor = t.stack(violations, dim=1)
    abs_tensor = t.stack(abs_values, dim=1)
    return violation_tensor, abs_tensor


def rmse_metric_dict(prefix: str, values: np.ndarray) -> dict:
    row = {}
    selected = []
    for label, idx in HORIZONS.items():
        value = float(values[idx])
        selected.append(value)
        row[f"{prefix}_rmse_{label}"] = value
    row[f"{prefix}_rmse_avg_4h"] = float(np.mean(selected))
    return row


def selection_metric(summary: dict) -> float:
    return float(
        np.mean(
            [
                float(summary["pos_rmse_avg_4h"]),
                float(summary["vel_rmse_avg_4h"]),
                float(summary["gap_rmse_avg_4h"]),
            ]
        )
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: str,
    split_name: str,
    variant: str,
    run_scope: str,
    selection_rule: str,
    alpha_pos: float,
    alpha_vel: float,
    alpha_gap: float,
    alpha_type: float,
    lambda_param: float,
    loss_mode: str,
    device_name: str,
    batch_size: int,
    gamma: float,
    max_batches: int | None = None,
) -> dict:
    device = t.device(device_name)
    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False
    run_args["gamma"] = gamma
    run_args["batch_size"] = batch_size
    run_args["aggregation_mode"] = "top2"
    run_args["route_top_k"] = 2

    encoder = model.Encoder(run_args).to(device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    simulator = model.predictor(run_args)

    out_length = int(run_args["out_length"])
    pos_acc = RMSEAccumulator(out_length, device)
    vel_acc = RMSEAccumulator(out_length, device)
    gap_acc = RMSEAccumulator(out_length, device)

    type_correct = np.zeros(int(run_args["cf_type"]), dtype=np.int64)
    type_total = np.zeros(int(run_args["cf_type"]), dtype=np.int64)
    param_loss_sum = 0.0
    param_loss_batches = 0
    param_violation_count = 0
    param_value_count = 0
    param_abs_violation_sum = 0.0
    negative_gap_count = 0
    dynamic_value_count = 0
    accel_bound_violation_count = 0
    accel_value_count = 0
    mean_abs_acc_sum = 0.0
    mean_abs_jerk_sum = 0.0
    jerk_value_count = 0
    sample_count = 0

    data = np.load(data_path, mmap_mode="r")
    usable = len(data) if max_batches is None else min(len(data), max_batches * batch_size)
    total_batches = (usable + batch_size - 1) // batch_size
    del data

    with t.inference_mode():
        for hist_np, nextv_np, fut_np, target_np in tqdm(
            make_batch_iterator(
                data_path,
                batch_size,
                run_args["in_length"],
                run_args["out_length"],
                max_batches,
            ),
            total=total_batches,
            desc=f"eval {variant} {split_name}",
        ):
            hist = t.as_tensor(hist_np, device=device)
            nextv = t.as_tensor(nextv_np, device=device)
            fut = t.as_tensor(fut_np, device=device)
            target = t.as_tensor(target_np, dtype=t.long, device=device)
            veh_state = hist[:, -1, [4, 2]]

            outputs = encoder(hist)
            predictions = simulator.forward(outputs, nextv, veh_state)
            direct_pred = predictions["direct_pred"].squeeze(-1)
            dynamic_pred = predictions["dynamic_pred"]
            params = outputs["params"]
            cf_probs = params["cf_probs"]
            route_probs = simulator.route_probabilities(cf_probs)

            pos_acc.update(direct_pred, fut[:, :, 2])
            gap_acc.update(dynamic_pred[:, :, 0], fut[:, :, 0])
            vel_acc.update(dynamic_pred[:, :, 1], fut[:, :, 1])

            pred_type = t.argmax(outputs["cf_type_pred"], dim=1)
            for type_id in TYPE_LABELS:
                mask = target == type_id
                type_total[type_id] += int(mask.sum().item())
                type_correct[type_id] += int(((pred_type == type_id) & mask).sum().item())

            batch_param_loss = param_distribution_loss(
                params["acc_params"], params["idm_params"], route_probs
            )
            param_loss_sum += float(batch_param_loss.item())
            param_loss_batches += 1

            violations, abs_violations = param_violation_diagnostics(
                params["acc_params"], params["idm_params"]
            )
            param_violation_count += int(violations.sum().item())
            param_value_count += int(violations.numel())
            param_abs_violation_sum += float(abs_violations.sum().item())

            negative_gap_count += int((dynamic_pred[:, :, 0] < 0).sum().item())
            dynamic_value_count += int(dynamic_pred[:, :, 0].numel())

            state = veh_state
            acc_steps = []
            for step in range(out_length):
                all_acc = compute_candidate_accelerations(
                    simulator, params, state, nextv[:, step : step + 1]
                )
                a_t = t.bmm(route_probs.unsqueeze(1), all_acc).squeeze(1)
                acc_steps.append(a_t)
                accel_bound_violation_count += int((a_t.abs() > 5.0 + 1e-6).sum().item())
                accel_value_count += int(a_t.numel())
                mean_abs_acc_sum += float(a_t.abs().sum().item())
                delta_v = nextv[:, step : step + 1] - state[:, 1:2]
                v_next = t.clamp(state[:, 1:2] + a_t * float(run_args["time_step"]), min=0)
                h_next = t.clamp(state[:, 0:1] + delta_v * float(run_args["time_step"]), min=0)
                state = t.cat([h_next, v_next], dim=-1)
            if len(acc_steps) > 1:
                acc_tensor = t.stack(acc_steps, dim=1)
                jerk = acc_tensor[:, 1:, :] - acc_tensor[:, :-1, :]
                mean_abs_jerk_sum += float(jerk.abs().sum().item())
                jerk_value_count += int(jerk.numel())

            sample_count += int(hist.shape[0])

    pos_values = pos_acc.result()
    vel_values = vel_acc.result()
    gap_values = gap_acc.result()
    row = {
        "variant": variant,
        "routing_mode": "top2",
        "loss_mode": loss_mode,
        "alpha_pos": alpha_pos,
        "alpha_vel": alpha_vel,
        "alpha_gap": alpha_gap,
        "alpha_type": alpha_type,
        "lambda_param": lambda_param,
        "run_scope": run_scope,
        "selection_rule": selection_rule,
        "selected_epoch": parse_epoch_from_checkpoint(checkpoint_path),
        "sample_count": sample_count,
        "checkpoint_path": str(checkpoint_path),
        "param_reg_loss": param_loss_sum / max(param_loss_batches, 1),
        "param_violation_rate": param_violation_count / max(param_value_count, 1),
        "param_violation_abs_mean": param_abs_violation_sum / max(param_value_count, 1),
        "negative_gap_rate": negative_gap_count / max(dynamic_value_count, 1),
        "acceleration_bound_violation_rate": accel_bound_violation_count
        / max(accel_value_count, 1),
        "mean_abs_acceleration": mean_abs_acc_sum / max(accel_value_count, 1),
        "mean_abs_jerk": mean_abs_jerk_sum / max(jerk_value_count, 1),
    }
    row.update(rmse_metric_dict("pos", pos_values))
    row.update(rmse_metric_dict("vel", vel_values))
    row.update(rmse_metric_dict("gap", gap_values))
    acc_values = []
    for type_id, label in TYPE_LABELS.items():
        acc = type_correct[type_id] / type_total[type_id] if type_total[type_id] else 0.0
        acc_values.append(acc)
        row[f"type_acc_{label.replace('-', '_')}"] = acc
    row["type_acc_avg"] = float(np.mean(acc_values))
    row["selection_metric"] = selection_metric(row)
    return row


def train_one(cli: argparse.Namespace) -> None:
    ensure_dirs()
    if cli.run_scope == "full" and cli.variant_name in completed_variants():
        print(f"Skipping completed variant: {cli.variant_name}")
        return
    if (
        FAST_REQUIRED_ONLY_SENTINEL.exists()
        and cli.run_scope == "full"
        and cli.variant_name not in REQUIRED_FULL_VARIANTS
    ):
        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
            ("variant",),
            {
                "variant": cli.variant_name,
                "alpha_pos": cli.alpha_pos,
                "alpha_vel": cli.alpha_vel,
                "alpha_gap": cli.alpha_gap,
                "alpha_type": cli.alpha_type,
                "lambda_param": cli.lambda_param,
                "loss_mode": cli.loss_mode,
                "routing_mode": "top2",
                "seed": cli.seed,
                "python_random_seed": cli.python_seed,
                "run_scope": cli.run_scope,
                "status": "skipped_optional_to_save_time",
                "selection_rule": "not run; optional variant skipped after runtime review",
                "checkpoint_dir": str(Path(cli.checkpoint_dir or variant_checkpoint_dir(cli.variant_name))),
                "checkpoint_path": "",
                "result_dir": str(Path(cli.result_dir or variant_result_dir(cli.variant_name))),
                "command": " ".join(sys.argv),
                "notes": "Skipped by FAST_REQUIRED_ONLY sentinel; minimum full grid retained.",
            },
            RUN_CONFIG_FIELDS,
        )
        write_report()
        print(
            f"Skipping optional variant {cli.variant_name} because "
            f"{FAST_REQUIRED_ONLY_SENTINEL} exists."
        )
        return

    lock_path = acquire_variant_lock(cli.variant_name)
    if lock_path is None:
        return

    set_seeds(cli.seed, cli.python_seed)
    if t.cuda.is_available() and t.device(cli.device).type == "cuda":
        t.set_float32_matmul_precision("high")

    run_args = base_run_args(cli, train_flag=True)
    checkpoint_dir = Path(cli.checkpoint_dir or variant_checkpoint_dir(cli.variant_name))
    result_dir = Path(cli.result_dir or variant_result_dir(cli.variant_name))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "variant": cli.variant_name,
        "alpha_pos": cli.alpha_pos,
        "alpha_vel": cli.alpha_vel,
        "alpha_gap": cli.alpha_gap,
        "alpha_type": cli.alpha_type,
        "lambda_param": cli.lambda_param,
        "loss_mode": cli.loss_mode,
        "routing_mode": "top2",
        "seed": cli.seed,
        "python_random_seed": cli.python_seed,
        "run_scope": cli.run_scope,
        "status": "running",
        "selection_rule": "best validation mean(pos,vel,gap avg_4h)",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": "",
        "result_dir": str(result_dir),
        "command": " ".join(sys.argv),
        "notes": "New sensitivity training output; does not overwrite default Top-2 run.",
    }
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
        ("variant",),
        run_config,
        RUN_CONFIG_FIELDS,
    )

    loss_scales = None
    if cli.loss_mode == "normalized":
        loss_scales = load_loss_scales(
            ANALYSIS_DIR / "loss_scale_statistics.csv", cli.train_data
        )

    encoder = model.Encoder(run_args)
    initialize_expert_biases(encoder)
    simulator = model.predictor(run_args)
    device = t.device(cli.device)
    encoder = encoder.to(device)
    encoder.train()

    train_dataset = None
    train_loader = None
    train_sample_count = int(len(np.load(cli.train_data, mmap_mode="r")))
    train_batches = train_sample_count // cli.batch_size
    if cli.max_batches is not None:
        train_batches = min(train_batches, cli.max_batches)
    if cli.train_loader == "dataloader":
        train_dataset = HighSimMemmapDataset(
            cli.train_data, run_args["in_length"], run_args["out_length"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cli.batch_size,
            shuffle=True,
            num_workers=cli.num_workers,
            pin_memory=t.cuda.is_available() and device.type == "cuda",
            drop_last=True,
        )
    optimizer = optim.Adam(encoder.parameters(), lr=cli.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cli.epochs, last_epoch=-1)

    if cli.last_epoch != 0:
        checkpoint = t.load(checkpoint_dir / f"epoch{cli.last_epoch}_e.tar", map_location=device)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Device: {device}")
    print("Aggregation: top2 k=2")
    print(f"Variant: {cli.variant_name}")
    print(f"Loss mode: {cli.loss_mode}")
    print(
        "Weights: "
        f"alpha_pos={cli.alpha_pos} alpha_vel={cli.alpha_vel} "
        f"alpha_gap={cli.alpha_gap} alpha_type={cli.alpha_type} "
        f"lambda_param={cli.lambda_param}"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Result dir: {result_dir}")
    print(f"Train samples: {train_sample_count}")
    print(f"Train loader: {cli.train_loader}")
    print(f"Trainable parameters: {count_parameters(encoder):,}")

    best_metric = float("inf")
    best_checkpoint = None
    existing_val_rows = read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_val_summary.csv")
    for row in existing_val_rows:
        if row.get("variant") == cli.variant_name and row.get("checkpoint_path"):
            try:
                best_metric = float(row.get("selection_metric", "inf"))
                best_checkpoint = Path(row["checkpoint_path"])
                print(
                    f"Resuming validation selection from {best_checkpoint} "
                    f"metric={best_metric:.6g}"
                )
            except ValueError:
                pass
            break

    for epoch in range(cli.last_epoch, cli.epochs):
        epoch_losses = {
            "pos": 0.0,
            "gap": 0.0,
            "vel": 0.0,
            "type": 0.0,
            "param": 0.0,
            "total": 0.0,
        }
        batch_count = 0
        skipped_nan_batches = 0
        print(f"epoch: {epoch + 1} lr {optimizer.param_groups[0]['lr']}")
        if cli.train_loader == "vectorized":
            train_iter = make_train_batch_iterator(
                cli.train_data,
                cli.batch_size,
                run_args["in_length"],
                run_args["out_length"],
                epoch,
                cli.seed,
                cli.max_batches,
            )
        else:
            train_iter = train_loader

        for data in tqdm(
            train_iter,
            total=train_batches,
            desc=f"train {cli.variant_name} e{epoch + 1}",
        ):
            hist, nextv, fut, cf_type = data
            hist = t.as_tensor(hist, device=device) if cli.train_loader == "vectorized" else hist.to(device, non_blocking=True)
            nextv = t.as_tensor(nextv, device=device) if cli.train_loader == "vectorized" else nextv.to(device, non_blocking=True)
            fut = t.as_tensor(fut, device=device) if cli.train_loader == "vectorized" else fut.to(device, non_blocking=True)
            cf_type = t.as_tensor(cf_type, device=device) if cli.train_loader == "vectorized" else cf_type.to(device, non_blocking=True)
            veh_state = hist[:, -1, [4, 2]]

            outputs = encoder(hist)
            predictions = simulator.forward(outputs, nextv, veh_state)
            direct_pred = predictions["direct_pred"]
            dynamic_pred = predictions["dynamic_pred"]
            params = outputs["params"]
            route_probs = simulator.route_probabilities(params["cf_probs"])

            if cli.loss_mode == "normalized":
                pos_error = (direct_pred.squeeze(-1) - fut[:, :, 2]) / loss_scales["sigma_pos"]
                gap_error = (dynamic_pred[:, :, 0] - fut[:, :, 0]) / loss_scales["sigma_gap"]
                vel_error = (dynamic_pred[:, :, 1] - fut[:, :, 1]) / loss_scales["sigma_vel"]
                direct_pos_loss = t.mean(pos_error.pow(2))
                dynamic_gap_loss = t.mean(gap_error.pow(2))
                dynamic_vel_loss = t.mean(vel_error.pow(2))
            else:
                direct_pos_loss = t.nn.MSELoss()(direct_pred.squeeze(-1), fut[:, :, 2])
                dynamic_gap_loss = t.nn.MSELoss()(dynamic_pred[:, :, 0], fut[:, :, 0])
                dynamic_vel_loss = t.nn.MSELoss()(dynamic_pred[:, :, 1], fut[:, :, 1])

            cf_type_loss = t.nn.CrossEntropyLoss()(outputs["cf_type_pred"], cf_type)
            param_dist_loss = param_distribution_loss(
                params["acc_params"], params["idm_params"], route_probs
            )
            total_loss = (
                cli.alpha_pos * direct_pos_loss
                + cli.alpha_vel * dynamic_vel_loss
                + cli.alpha_gap * dynamic_gap_loss
                + cli.alpha_type * cf_type_loss
                + cli.lambda_param * param_dist_loss
            )
            if not t.isfinite(total_loss).all():
                skipped_nan_batches += 1
                if skipped_nan_batches <= cli.max_nan_batches:
                    print(
                        "WARNING: non-finite loss skipped "
                        f"variant={cli.variant_name} epoch={epoch + 1} "
                        f"batch={batch_count + 1} total_loss={total_loss.item()}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise RuntimeError(
                    "NaN detected in total loss after exceeding "
                    f"--max-nan-batches={cli.max_nan_batches}."
                )
            optimizer.zero_grad()
            total_loss.backward()
            t.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_losses["pos"] += float(direct_pos_loss.item())
            epoch_losses["gap"] += float(dynamic_gap_loss.item())
            epoch_losses["vel"] += float(dynamic_vel_loss.item())
            epoch_losses["type"] += float(cf_type_loss.item())
            epoch_losses["param"] += float(param_dist_loss.item())
            epoch_losses["total"] += float(total_loss.item())
            batch_count += 1
            if (
                cli.train_loader == "dataloader"
                and cli.max_batches is not None
                and batch_count >= cli.max_batches
            ):
                break

        scheduler.step()
        avg = {name: value / max(batch_count, 1) for name, value in epoch_losses.items()}
        checkpoint_path = checkpoint_dir / f"epoch{epoch + 1}_e.tar"
        t.save(
            {
                "epoch": epoch,
                "model_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg["total"],
                "aggregation_mode": "top2",
                "route_top_k": 2,
                "alpha_pos": cli.alpha_pos,
                "alpha_vel": cli.alpha_vel,
                "alpha_gap": cli.alpha_gap,
                "alpha_type": cli.alpha_type,
                "lambda_param": cli.lambda_param,
                "loss_mode": cli.loss_mode,
                "seed": cli.seed,
                "python_random_seed": cli.python_seed,
            },
            checkpoint_path,
        )

        should_validate = ((epoch + 1) % cli.val_every == 0) or (epoch + 1 == cli.epochs)
        val_summary = None
        is_best = False
        if should_validate:
            val_summary = evaluate_checkpoint(
                checkpoint_path,
                cli.valid_data,
                "valid",
                cli.variant_name,
                cli.run_scope,
                f"best validation mean(pos,vel,gap avg_4h), evaluated every {cli.val_every} epoch(s) and final",
                cli.alpha_pos,
                cli.alpha_vel,
                cli.alpha_gap,
                cli.alpha_type,
                cli.lambda_param,
                cli.loss_mode,
                cli.device,
                cli.eval_batch_size,
                cli.gamma,
                cli.max_eval_batches,
            )
            is_best = val_summary["selection_metric"] < best_metric
            if is_best:
                best_metric = val_summary["selection_metric"]
                best_checkpoint = checkpoint_path

        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_train_log.csv",
            ("variant", "epoch"),
            {
                "variant": cli.variant_name,
                "epoch": epoch + 1,
                "run_scope": cli.run_scope,
                "loss_mode": cli.loss_mode,
                "routing_mode": "top2",
                "alpha_pos": cli.alpha_pos,
                "alpha_vel": cli.alpha_vel,
                "alpha_gap": cli.alpha_gap,
                "alpha_type": cli.alpha_type,
                "lambda_param": cli.lambda_param,
                "train_total_loss": avg["total"],
                "train_pos_loss": avg["pos"],
                "train_vel_loss": avg["vel"],
                "train_gap_loss": avg["gap"],
                "train_type_loss": avg["type"],
                "train_param_loss": avg["param"],
                "validation_selection_metric": "" if val_summary is None else val_summary["selection_metric"],
                "val_pos_rmse_0.5s": "" if val_summary is None else val_summary["pos_rmse_0.5s"],
                "val_pos_rmse_1.0s": "" if val_summary is None else val_summary["pos_rmse_1.0s"],
                "val_pos_rmse_1.5s": "" if val_summary is None else val_summary["pos_rmse_1.5s"],
                "val_pos_rmse_2.0s": "" if val_summary is None else val_summary["pos_rmse_2.0s"],
                "val_pos_rmse_avg_4h": "" if val_summary is None else val_summary["pos_rmse_avg_4h"],
                "val_vel_rmse_0.5s": "" if val_summary is None else val_summary["vel_rmse_0.5s"],
                "val_vel_rmse_1.0s": "" if val_summary is None else val_summary["vel_rmse_1.0s"],
                "val_vel_rmse_1.5s": "" if val_summary is None else val_summary["vel_rmse_1.5s"],
                "val_vel_rmse_2.0s": "" if val_summary is None else val_summary["vel_rmse_2.0s"],
                "val_vel_rmse_avg_4h": "" if val_summary is None else val_summary["vel_rmse_avg_4h"],
                "val_gap_rmse_0.5s": "" if val_summary is None else val_summary["gap_rmse_0.5s"],
                "val_gap_rmse_1.0s": "" if val_summary is None else val_summary["gap_rmse_1.0s"],
                "val_gap_rmse_1.5s": "" if val_summary is None else val_summary["gap_rmse_1.5s"],
                "val_gap_rmse_2.0s": "" if val_summary is None else val_summary["gap_rmse_2.0s"],
                "val_gap_rmse_avg_4h": "" if val_summary is None else val_summary["gap_rmse_avg_4h"],
                "val_type_acc_avg": "" if val_summary is None else val_summary["type_acc_avg"],
                "checkpoint_path": str(checkpoint_path),
                "is_best_checkpoint": int(is_best),
                "skipped_nan_batches": skipped_nan_batches,
            },
            TRAIN_LOG_FIELDS,
        )
        if is_best and val_summary is not None:
            upsert_row(
                ANALYSIS_DIR / "loss_sensitivity_val_summary.csv",
                ("variant",),
                val_summary,
                SUMMARY_FIELDS,
            )
            run_config["checkpoint_path"] = str(best_checkpoint)
            run_config["status"] = f"running_epoch_{epoch + 1}_best_so_far"
            upsert_row(
                ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
                ("variant",),
                run_config,
                RUN_CONFIG_FIELDS,
            )

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was selected.")

    test_summary = evaluate_checkpoint(
        best_checkpoint,
        cli.test_data,
        "test",
        cli.variant_name,
        cli.run_scope,
        "best validation mean(pos,vel,gap avg_4h)",
        cli.alpha_pos,
        cli.alpha_vel,
        cli.alpha_gap,
        cli.alpha_type,
        cli.lambda_param,
        cli.loss_mode,
        cli.device,
        cli.eval_batch_size,
        cli.gamma,
        cli.max_eval_batches if cli.run_scope == "pilot" else None,
    )
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_test_summary.csv",
        ("variant",),
        test_summary,
        SUMMARY_FIELDS,
    )
    run_config["status"] = "completed"
    run_config["checkpoint_path"] = str(best_checkpoint)
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
        ("variant",),
        run_config,
        RUN_CONFIG_FIELDS,
    )
    write_report()


def wait_for_checkpoint(checkpoint_path: Path, poll_seconds: int, stable_seconds: int) -> None:
    print(f"Waiting for default checkpoint: {checkpoint_path}")
    while not checkpoint_path.exists():
        time.sleep(poll_seconds)
    last_size = -1
    stable_start = None
    while True:
        size = checkpoint_path.stat().st_size
        if size == last_size and size > 0:
            if stable_start is None:
                stable_start = time.time()
            if time.time() - stable_start >= stable_seconds:
                break
        else:
            last_size = size
            stable_start = None
        time.sleep(min(poll_seconds, 30))
    print(f"Checkpoint is present and stable: {checkpoint_path}")


def list_epoch_checkpoints(checkpoint_dir: Path, max_epoch: int) -> list[Path]:
    paths = []
    for epoch in range(1, max_epoch + 1):
        path = checkpoint_dir / f"epoch{epoch}_e.tar"
        if path.exists():
            paths.append(path)
    return paths


def eval_existing_default(cli: argparse.Namespace) -> None:
    ensure_dirs()
    variant = "default_raw"
    checkpoint_dir = Path(cli.default_checkpoint_dir)
    final_checkpoint = checkpoint_dir / f"epoch{cli.epochs}_e.tar"
    if cli.wait_for_default:
        wait_for_checkpoint(final_checkpoint, cli.poll_seconds, cli.stable_seconds)
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"Default final checkpoint not found: {final_checkpoint}")

    run_config = {
        "variant": variant,
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
        "routing_mode": "top2",
        "seed": cli.seed,
        "python_random_seed": cli.python_seed,
        "run_scope": "full",
        "status": "running_inference_only",
        "selection_rule": cli.default_selection,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": "",
        "result_dir": str(DEFAULT_TOP2_RESULT_DIR),
        "command": " ".join(sys.argv),
        "notes": "Reuses the ongoing/default Top-2 training output; no retraining is performed.",
    }
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
        ("variant",),
        run_config,
        RUN_CONFIG_FIELDS,
    )

    if cli.default_selection == "final":
        candidates = [final_checkpoint]
        selection_rule = "final checkpoint inference only"
    else:
        candidates = list_epoch_checkpoints(checkpoint_dir, cli.epochs)
        selection_rule = "best validation mean(pos,vel,gap avg_4h) over existing checkpoints"

    best_metric = float("inf")
    best_checkpoint = None
    best_val = None
    for checkpoint_path in candidates:
        val_summary = evaluate_checkpoint(
            checkpoint_path,
            cli.valid_data,
            "valid",
            variant,
            "full",
            selection_rule,
            1.0,
            1.0,
            1.0,
            1.0,
            0.01,
            "raw",
            cli.device,
            cli.eval_batch_size,
            cli.gamma,
            cli.max_eval_batches,
        )
        metric = float(val_summary["selection_metric"])
        is_best = metric < best_metric
        if is_best:
            best_metric = metric
            best_checkpoint = checkpoint_path
            best_val = val_summary
        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_train_log.csv",
            ("variant", "epoch"),
            {
                "variant": variant,
                "epoch": parse_epoch_from_checkpoint(checkpoint_path),
                "run_scope": "full",
                "loss_mode": "raw",
                "routing_mode": "top2",
                "alpha_pos": 1.0,
                "alpha_vel": 1.0,
                "alpha_gap": 1.0,
                "alpha_type": 1.0,
                "lambda_param": 0.01,
                "validation_selection_metric": metric,
                "val_pos_rmse_0.5s": val_summary["pos_rmse_0.5s"],
                "val_pos_rmse_1.0s": val_summary["pos_rmse_1.0s"],
                "val_pos_rmse_1.5s": val_summary["pos_rmse_1.5s"],
                "val_pos_rmse_2.0s": val_summary["pos_rmse_2.0s"],
                "val_pos_rmse_avg_4h": val_summary["pos_rmse_avg_4h"],
                "val_vel_rmse_0.5s": val_summary["vel_rmse_0.5s"],
                "val_vel_rmse_1.0s": val_summary["vel_rmse_1.0s"],
                "val_vel_rmse_1.5s": val_summary["vel_rmse_1.5s"],
                "val_vel_rmse_2.0s": val_summary["vel_rmse_2.0s"],
                "val_vel_rmse_avg_4h": val_summary["vel_rmse_avg_4h"],
                "val_gap_rmse_0.5s": val_summary["gap_rmse_0.5s"],
                "val_gap_rmse_1.0s": val_summary["gap_rmse_1.0s"],
                "val_gap_rmse_1.5s": val_summary["gap_rmse_1.5s"],
                "val_gap_rmse_2.0s": val_summary["gap_rmse_2.0s"],
                "val_gap_rmse_avg_4h": val_summary["gap_rmse_avg_4h"],
                "val_type_acc_avg": val_summary["type_acc_avg"],
                "checkpoint_path": str(checkpoint_path),
                "is_best_checkpoint": int(is_best),
                "skipped_nan_batches": "",
            },
            TRAIN_LOG_FIELDS,
        )

    if best_checkpoint is None or best_val is None:
        raise RuntimeError("No default checkpoint could be evaluated.")
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_val_summary.csv",
        ("variant",),
        best_val,
        SUMMARY_FIELDS,
    )
    test_summary = evaluate_checkpoint(
        best_checkpoint,
        cli.test_data,
        "test",
        variant,
        "full",
        selection_rule,
        1.0,
        1.0,
        1.0,
        1.0,
        0.01,
        "raw",
        cli.device,
        cli.eval_batch_size,
        cli.gamma,
        None,
    )
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_test_summary.csv",
        ("variant",),
        test_summary,
        SUMMARY_FIELDS,
    )
    run_config["status"] = "completed"
    run_config["checkpoint_path"] = str(best_checkpoint)
    run_config["selection_rule"] = selection_rule
    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
        ("variant",),
        run_config,
        RUN_CONFIG_FIELDS,
    )
    write_report()


def command_for_train(variant: dict, cli: argparse.Namespace, run_scope: str = "full") -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "loss_sensitivity_top2.py",
        "train",
        "--variant-name",
        variant["variant"],
        "--alpha-pos",
        str(variant["alpha_pos"]),
        "--alpha-vel",
        str(variant["alpha_vel"]),
        "--alpha-gap",
        str(variant["alpha_gap"]),
        "--alpha-type",
        str(variant["alpha_type"]),
        "--lambda-param",
        str(variant["lambda_param"]),
        "--loss-mode",
        variant["loss_mode"],
        "--run-scope",
        run_scope,
        "--epochs",
        str(cli.epochs),
        "--batch-size",
        str(cli.batch_size),
        "--eval-batch-size",
        str(cli.eval_batch_size),
        "--num-workers",
        str(cli.num_workers),
        "--gamma",
        str(cli.gamma),
        "--device",
        cli.device,
        "--seed",
        str(cli.seed),
        "--python-seed",
        str(cli.python_seed),
        "--val-every",
        str(cli.val_every),
        "--train-loader",
        cli.train_loader,
        "--max-nan-batches",
        str(cli.max_nan_batches),
    ]
    resume_epoch = latest_epoch_checkpoint(variant_checkpoint_dir(variant["variant"]))
    if resume_epoch > 0:
        cmd.extend(["--last-epoch", str(resume_epoch)])
    return cmd


def latest_epoch_checkpoint(checkpoint_dir: Path) -> int:
    latest = 0
    if not checkpoint_dir.exists():
        return latest
    for path in checkpoint_dir.glob("epoch*_e.tar"):
        epoch = parse_epoch_from_checkpoint(path)
        if isinstance(epoch, int):
            latest = max(latest, epoch)
    return latest


def completed_variants() -> set[str]:
    return {
        row.get("variant", "")
        for row in read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_test_summary.csv")
        if row.get("variant")
    }


def variant_by_name(name: str) -> dict | None:
    for variant in VARIANTS:
        if variant["variant"] == name:
            return variant
    return None


def finalize_failed_variant_from_best_checkpoint(
    variant: dict,
    cli: argparse.Namespace,
    failure_note: str,
) -> bool:
    variant_name = variant["variant"]
    val_rows = read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_val_summary.csv")
    best_row = next((row for row in val_rows if row.get("variant") == variant_name), None)
    if not best_row or not best_row.get("checkpoint_path"):
        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
            ("variant",),
            {
                "variant": variant_name,
                "alpha_pos": variant["alpha_pos"],
                "alpha_vel": variant["alpha_vel"],
                "alpha_gap": variant["alpha_gap"],
                "alpha_type": variant["alpha_type"],
                "lambda_param": variant["lambda_param"],
                "loss_mode": variant["loss_mode"],
                "routing_mode": "top2",
                "seed": cli.seed,
                "python_random_seed": cli.python_seed,
                "run_scope": "full",
                "status": "failed_no_valid_checkpoint",
                "selection_rule": "failed before a validation-selected checkpoint was available",
                "checkpoint_dir": str(variant_checkpoint_dir(variant_name)),
                "checkpoint_path": "",
                "result_dir": str(variant_result_dir(variant_name)),
                "command": "",
                "notes": failure_note,
            },
            RUN_CONFIG_FIELDS,
        )
        return False

    checkpoint_path = Path(best_row["checkpoint_path"])
    if not checkpoint_path.exists():
        return False

    if variant_name not in completed_variants():
        test_summary = evaluate_checkpoint(
            checkpoint_path,
            cli.test_data,
            "test",
            variant_name,
            "full",
            "best validation checkpoint before failed/non-finite continuation",
            variant["alpha_pos"],
            variant["alpha_vel"],
            variant["alpha_gap"],
            variant["alpha_type"],
            variant["lambda_param"],
            variant["loss_mode"],
            cli.device,
            cli.eval_batch_size,
            cli.gamma,
            None,
        )
        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_test_summary.csv",
            ("variant",),
            test_summary,
            SUMMARY_FIELDS,
        )

    upsert_row(
        ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
        ("variant",),
        {
            "variant": variant_name,
            "alpha_pos": variant["alpha_pos"],
            "alpha_vel": variant["alpha_vel"],
            "alpha_gap": variant["alpha_gap"],
            "alpha_type": variant["alpha_type"],
            "lambda_param": variant["lambda_param"],
            "loss_mode": variant["loss_mode"],
            "routing_mode": "top2",
            "seed": cli.seed,
            "python_random_seed": cli.python_seed,
            "run_scope": "full",
            "status": "stopped_after_nonfinite_or_failed_continuation",
            "selection_rule": "used best validation checkpoint available before failure",
            "checkpoint_dir": str(variant_checkpoint_dir(variant_name)),
            "checkpoint_path": str(checkpoint_path),
            "result_dir": str(variant_result_dir(variant_name)),
            "command": "",
            "notes": failure_note,
        },
        RUN_CONFIG_FIELDS,
    )
    write_report()
    return True


def finalize_failed_cli(cli: argparse.Namespace) -> None:
    variant = variant_by_name(cli.variant_name)
    if variant is None:
        raise ValueError(f"Unknown variant: {cli.variant_name}")
    ok = finalize_failed_variant_from_best_checkpoint(variant, cli, cli.note)
    if not ok:
        raise RuntimeError(
            f"No validation-selected checkpoint available for {cli.variant_name}."
        )


def write_command_list(cli: argparse.Namespace) -> None:
    lines = [
        "# Loss Sensitivity Top-2 Reproduction Commands",
        "",
        "Run from the repository root. These commands do not overwrite the ongoing default Top-2 output.",
        "",
        "## Default Raw Evaluation Only",
        " ".join(
            [
                sys.executable,
                "-u",
                "loss_sensitivity_top2.py",
                "eval-default",
                "--wait-for-default",
                "--default-selection",
                cli.default_selection,
                "--device",
                cli.device,
            ]
        ),
        "",
        "## New Sensitivity Training Runs",
    ]
    for variant in VARIANTS:
        if variant["reuse_default"]:
            continue
        lines.append(" ".join(command_for_train(variant, cli)))
    lines.extend(
        [
            "",
            "## Regenerate Report",
            f"{sys.executable} -u loss_sensitivity_top2.py report",
        ]
    )
    (ANALYSIS_DIR / "commands_to_complete.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def initialize_material_tables(cli: argparse.Namespace) -> None:
    for path, fields in [
        (ANALYSIS_DIR / "loss_sensitivity_run_config.csv", RUN_CONFIG_FIELDS),
        (ANALYSIS_DIR / "loss_sensitivity_train_log.csv", TRAIN_LOG_FIELDS),
        (ANALYSIS_DIR / "loss_sensitivity_val_summary.csv", SUMMARY_FIELDS),
        (ANALYSIS_DIR / "loss_sensitivity_test_summary.csv", SUMMARY_FIELDS),
    ]:
        if not path.exists():
            write_csv_rows(path, [], fields)

    existing = {
        row.get("variant")
        for row in read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_run_config.csv")
    }
    default_command = " ".join(
        [
            sys.executable,
            "-u",
            "loss_sensitivity_top2.py",
            "eval-default",
            "--wait-for-default",
            "--default-selection",
            cli.default_selection,
            "--device",
            cli.device,
        ]
    )
    for variant in VARIANTS:
        name = variant["variant"]
        if name in existing:
            continue
        if variant["reuse_default"]:
            checkpoint_dir = str(DEFAULT_TOP2_CHECKPOINT_DIR)
            result_dir = str(DEFAULT_TOP2_RESULT_DIR)
            status = "pending_default_inference_only"
            command = default_command
            notes = "Reuses the ongoing/default Top-2 training output; no retraining is performed."
        else:
            checkpoint_dir = str(variant_checkpoint_dir(name))
            result_dir = str(variant_result_dir(name))
            status = "pending_training"
            command = " ".join(command_for_train(variant, cli))
            notes = "New sensitivity training output; isolated from the default Top-2 run."
        upsert_row(
            ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
            ("variant",),
            {
                "variant": name,
                "alpha_pos": variant["alpha_pos"],
                "alpha_vel": variant["alpha_vel"],
                "alpha_gap": variant["alpha_gap"],
                "alpha_type": variant["alpha_type"],
                "lambda_param": variant["lambda_param"],
                "loss_mode": variant["loss_mode"],
                "routing_mode": "top2",
                "seed": cli.seed,
                "python_random_seed": cli.python_seed,
                "run_scope": "full",
                "status": status,
                "selection_rule": "best validation mean(pos,vel,gap avg_4h)",
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_path": "",
                "result_dir": result_dir,
                "command": command,
                "notes": notes,
            },
            RUN_CONFIG_FIELDS,
        )


def run_pilot(cli: argparse.Namespace) -> None:
    pilot_variant = {
        "variant": "_pilot_smoke_default_raw",
        "alpha_pos": 1.0,
        "alpha_vel": 1.0,
        "alpha_gap": 1.0,
        "alpha_type": 1.0,
        "lambda_param": 0.01,
        "loss_mode": "raw",
    }
    cmd = command_for_train(pilot_variant, cli, run_scope="pilot") + [
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--eval-batch-size",
        "256",
        "--max-batches",
        "2",
        "--max-eval-batches",
        "1",
    ]
    print("Running pilot smoke check:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_grid(cli: argparse.Namespace) -> None:
    ensure_dirs()
    save_implementation_check()
    write_command_list(cli)
    initialize_material_tables(cli)

    if cli.reuse_default:
        cmd = [
            sys.executable,
            "-u",
            "loss_sensitivity_top2.py",
            "eval-default",
            "--default-checkpoint-dir",
            str(DEFAULT_TOP2_CHECKPOINT_DIR),
            "--default-selection",
            cli.default_selection,
            "--epochs",
            str(cli.epochs),
            "--eval-batch-size",
            str(cli.eval_batch_size),
            "--device",
            cli.device,
            "--gamma",
            str(cli.gamma),
            "--seed",
            str(cli.seed),
            "--python-seed",
            str(cli.python_seed),
        ]
        if cli.wait_for_default:
            cmd.append("--wait-for-default")
        print("Evaluating existing default Top-2 run:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    if cli.compute_scales:
        load_loss_scales(ANALYSIS_DIR / "loss_scale_statistics.csv", cli.train_data)

    if cli.run_pilot:
        run_pilot(cli)

    done = completed_variants()
    for variant in VARIANTS:
        if variant["reuse_default"]:
            continue
        if variant["variant"] in done:
            print(f"Skipping completed variant: {variant['variant']}")
            continue
        if FAST_REQUIRED_ONLY_SENTINEL.exists() and variant["variant"] not in REQUIRED_FULL_VARIANTS:
            upsert_row(
                ANALYSIS_DIR / "loss_sensitivity_run_config.csv",
                ("variant",),
                {
                    "variant": variant["variant"],
                    "alpha_pos": variant["alpha_pos"],
                    "alpha_vel": variant["alpha_vel"],
                    "alpha_gap": variant["alpha_gap"],
                    "alpha_type": variant["alpha_type"],
                    "lambda_param": variant["lambda_param"],
                    "loss_mode": variant["loss_mode"],
                    "routing_mode": "top2",
                    "seed": cli.seed,
                    "python_random_seed": cli.python_seed,
                    "run_scope": "full",
                    "status": "skipped_optional_to_save_time",
                    "selection_rule": "not run; optional variant skipped after runtime review",
                    "checkpoint_dir": str(variant_checkpoint_dir(variant["variant"])),
                    "checkpoint_path": "",
                    "result_dir": str(variant_result_dir(variant["variant"])),
                    "command": "",
                    "notes": "Skipped by FAST_REQUIRED_ONLY sentinel; minimum full grid retained.",
                },
                RUN_CONFIG_FIELDS,
            )
            print(f"Skipping optional variant: {variant['variant']}")
            continue
        if cli.only and variant["variant"] not in cli.only:
            continue
        cmd = command_for_train(variant, cli, run_scope="full")
        print(f"Starting full sensitivity run: {variant['variant']}")
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            note = (
                f"Training subprocess exited with code {exc.returncode}. "
                "Using the best validation checkpoint already available for this variant, "
                "then continuing the remaining queue."
            )
            print(f"WARNING: {variant['variant']} failed. {note}")
            finalize_failed_variant_from_best_checkpoint(variant, cli, note)
            continue
        write_report()


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows available yet._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join(lines)


def write_report() -> None:
    ensure_dirs()
    run_rows = read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_run_config.csv")
    val_rows = read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_val_summary.csv")
    test_rows = read_csv_rows(ANALYSIS_DIR / "loss_sensitivity_test_summary.csv")

    run_rows_sorted = sorted(run_rows, key=lambda r: r.get("variant", ""))
    test_rows_sorted = sorted(test_rows, key=lambda r: r.get("variant", ""))
    variant_order = {variant["variant"]: idx for idx, variant in enumerate(VARIANTS)}

    completed_full_test_rows = sorted(
        [
            row
            for row in test_rows
            if row.get("run_scope") == "full"
            and row.get("checkpoint_path")
            and row.get("variant")
        ],
        key=lambda r: variant_order.get(r.get("variant", ""), 10_000),
    )
    completed_full_variants = {row.get("variant") for row in completed_full_test_rows}
    completed_full_run_rows = sorted(
        [
            row
            for row in run_rows
            if row.get("variant") in completed_full_variants
            and row.get("run_scope") == "full"
        ],
        key=lambda r: variant_order.get(r.get("variant", ""), 10_000),
    )
    completed_full_val_rows = sorted(
        [
            row
            for row in val_rows
            if row.get("variant") in completed_full_variants
            and row.get("run_scope") == "full"
        ],
        key=lambda r: variant_order.get(r.get("variant", ""), 10_000),
    )

    compact_test_rows = []
    for row in completed_full_test_rows:
        compact_test_rows.append(
            {
                "variant": row.get("variant", ""),
                "loss_mode": row.get("loss_mode", ""),
                "alpha_pos": row.get("alpha_pos", ""),
                "alpha_vel": row.get("alpha_vel", ""),
                "alpha_gap": row.get("alpha_gap", ""),
                "alpha_type": row.get("alpha_type", ""),
                "lambda_param": row.get("lambda_param", ""),
                "selected_epoch": row.get("selected_epoch", ""),
                "pos_rmse_2.0s": fmt_float(row.get("pos_rmse_2.0s")),
                "pos_rmse_avg_4h": fmt_float(row.get("pos_rmse_avg_4h")),
                "vel_rmse_2.0s": fmt_float(row.get("vel_rmse_2.0s")),
                "vel_rmse_avg_4h": fmt_float(row.get("vel_rmse_avg_4h")),
                "gap_rmse_2.0s": fmt_float(row.get("gap_rmse_2.0s")),
                "gap_rmse_avg_4h": fmt_float(row.get("gap_rmse_avg_4h")),
            }
        )
    compact_fields = [
        "variant",
        "loss_mode",
        "alpha_pos",
        "alpha_vel",
        "alpha_gap",
        "alpha_type",
        "lambda_param",
        "selected_epoch",
        "pos_rmse_2.0s",
        "pos_rmse_avg_4h",
        "vel_rmse_2.0s",
        "vel_rmse_avg_4h",
        "gap_rmse_2.0s",
        "gap_rmse_avg_4h",
    ]
    write_csv_rows(
        ANALYSIS_DIR / "loss_sensitivity_compact_test_table.csv",
        compact_test_rows,
        compact_fields,
    )

    completed = {r.get("variant") for r in test_rows}
    skipped = {
        r.get("variant")
        for r in run_rows
        if r.get("status") == "skipped_optional_to_save_time"
    }
    all_variants = {v["variant"] for v in VARIANTS}
    missing = sorted(all_variants - completed - skipped)
    skipped_missing = sorted((all_variants - completed) & skipped)

    strongest = []
    if completed_full_test_rows:
        numeric_fields = [
            "pos_rmse_avg_4h",
            "vel_rmse_avg_4h",
            "gap_rmse_avg_4h",
            "type_acc_avg",
            "param_reg_loss",
            "param_violation_rate",
        ]
        for field in numeric_fields:
            values = []
            for row in completed_full_test_rows:
                try:
                    values.append((float(row[field]), row["variant"]))
                except (KeyError, TypeError, ValueError):
                    pass
            if values:
                if field == "type_acc_avg":
                    best_value, best_variant = max(values)
                else:
                    best_value, best_variant = min(values)
                strongest.append(f"- Best `{field}` so far: `{best_variant}` = {best_value:.6g}.")

    report = f"""# Loss Sensitivity Top-2 Report

This report summarizes computed materials for the PreSimNet Top-2 loss-sensitivity experiments. It is not reviewer response text, not manuscript text, and not LaTeX.

## Purpose

Compare several multi-task loss-weight settings under the same revised Top-2 routing protocol while keeping the same data split, architecture, optimizer, scheduler, random seed, and metrics. The default Top-2 run is reused for inference only and is not retrained by this package.

## Confirmed Implementation

- Model file used by this package: `model/model_MoE_gru_new.py`.
- Legacy `train_new.py` imports the ablation model, but the active Top-2 training script `train_moe_top2.py` imports `model/model_MoE_gru_new.py`.
- Loss terms are direct position MSE, dynamic gap MSE, dynamic velocity MSE, type classification cross-entropy, and physical parameter-bound regularization.
- Top-2 routing uses predicted `cf_probs`, selects the two largest predicted probabilities, renormalizes over those two experts, and aggregates candidate accelerations only.
- ACC and IDM parameters are not mixed; each expert first produces its own acceleration.

## Data And Protocol

- Train split: `../data/train_data.npy`
- Validation split: `../data/valid_data.npy`
- Test split: `../data/test_data.npy`
- New outputs: `checkponint/loss_sensitivity_top2/` and `result/loss_sensitivity_top2/`
- Summary materials: `fig/vis/loss_sensitivity_top2/`
- Selection rule: best validation mean of position, velocity, and gap `avg_4h`; test is evaluated once for the selected checkpoint.
- `avg_4h`: mean of RMSE at 0.5s, 1.0s, 1.5s, and 2.0s.
- RMSE values in the validation and test summary tables use the legacy `evaluate_new.py` aggregation style with `batch_size = 512` and `drop_last = True`: each batch contributes `sqrt(sum squared error)` and `sqrt(batch count)`, matching the original result CSV convention.

## Completed Runs

{markdown_table(completed_full_run_rows, [
    ("Variant", "variant"),
    ("Mode", "loss_mode"),
    ("pos", "alpha_pos"),
    ("vel", "alpha_vel"),
    ("gap", "alpha_gap"),
    ("type", "alpha_type"),
    ("param", "lambda_param"),
    ("Scope", "run_scope"),
    ("Status", "status"),
])}

## Validation Summary

{markdown_table(completed_full_val_rows, [
    ("Variant", "variant"),
    ("Selected epoch", "selected_epoch"),
    ("Sel. metric", "selection_metric"),
    ("pos avg_4h", "pos_rmse_avg_4h"),
    ("vel avg_4h", "vel_rmse_avg_4h"),
    ("gap avg_4h", "gap_rmse_avg_4h"),
    ("type acc", "type_acc_avg"),
])}

## Test Metrics

This table includes completed full runs only. Position is open-loop direct prediction RMSE; velocity and gap are closed-loop simulation RMSE. RMSE values use the legacy `evaluate_new.py` batch-wise aggregation style with `batch_size = 512` and `drop_last = True`. `avg_4h` is the mean over 0.5s, 1.0s, 1.5s, and 2.0s.

{markdown_table(compact_test_rows, [
    ("Variant", "variant"),
    ("Mode", "loss_mode"),
    ("alpha_pos", "alpha_pos"),
    ("alpha_vel", "alpha_vel"),
    ("alpha_gap", "alpha_gap"),
    ("alpha_type", "alpha_type"),
    ("lambda_param", "lambda_param"),
    ("Epoch", "selected_epoch"),
    ("pos 2.0s", "pos_rmse_2.0s"),
    ("pos avg_4h", "pos_rmse_avg_4h"),
    ("vel 2.0s", "vel_rmse_2.0s"),
    ("vel avg_4h", "vel_rmse_avg_4h"),
    ("gap 2.0s", "gap_rmse_2.0s"),
    ("gap avg_4h", "gap_rmse_avg_4h"),
])}

## Raw Vs Normalized Loss

- `raw` variants use the original physical-unit MSE losses.
- `normalized` variants divide position, velocity, and gap errors by train-set-only scale estimates before squaring.
- Scale statistics are saved in `loss_scale_statistics.csv`.

## Strongest Quantitative Facts So Far

{chr(10).join(strongest) if strongest else "_No completed test summaries yet._"}

## Unfavorable Or Neutral Results

Inspect the test table for variants that do not improve over `default_raw`; these are intentionally retained rather than filtered out.

## Limitations And Pending Work

{("- Pending final runs: " + ", ".join(missing) + ".") if missing else "- All non-skipped planned full variants have completed test summaries."}
{("- Skipped by runtime/design decision: " + ", ".join(skipped_missing) + ".") if skipped_missing else ""}
- Pilot rows, if present, are marked with `run_scope = pilot` and should not be reported as final numbers.
- `default_raw` reuses the completed default Top-2 checkpoint for inference only; it is not retrained by this package.
- Commands to complete or reproduce the package are saved in `commands_to_complete.txt`.
"""
    (ANALYSIS_DIR / "loss_sensitivity_report.md").write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loss sensitivity experiments for revised Top-2 PreSimNet."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train-data", default="../data/train_data.npy")
    common.add_argument("--valid-data", default="../data/valid_data.npy")
    common.add_argument("--test-data", default="../data/test_data.npy")
    common.add_argument("--epochs", type=int, default=21)
    common.add_argument("--last-epoch", type=int, default=0)
    common.add_argument("--batch-size", type=int, default=512)
    common.add_argument("--eval-batch-size", type=int, default=4096)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--lr", type=float, default=learning_rate)
    common.add_argument("--gamma", type=float, default=0.9)
    common.add_argument("--seed", type=int, default=72)
    common.add_argument("--python-seed", type=int, default=30)
    common.add_argument("--device", default="cuda:0" if t.cuda.is_available() else "cpu")
    common.add_argument("--max-batches", type=int, default=None)
    common.add_argument("--max-eval-batches", type=int, default=None)
    common.add_argument("--val-every", type=int, default=7)
    common.add_argument("--max-nan-batches", type=int, default=0)
    common.add_argument(
        "--train-loader",
        choices=("vectorized", "dataloader"),
        default="vectorized",
        help=(
            "vectorized reads contiguous memmap batches and shuffles batch blocks; "
            "dataloader preserves the older per-sample DataLoader path."
        ),
    )

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--variant-name", required=True)
    train.add_argument("--alpha-pos", type=float, default=1.0)
    train.add_argument("--alpha-vel", type=float, default=1.0)
    train.add_argument("--alpha-gap", type=float, default=1.0)
    train.add_argument("--alpha-type", type=float, default=1.0)
    train.add_argument("--lambda-param", type=float, default=0.01)
    train.add_argument("--loss-mode", choices=("raw", "normalized"), default="raw")
    train.add_argument("--run-scope", choices=("full", "pilot"), default="full")
    train.add_argument("--checkpoint-dir", default=None)
    train.add_argument("--result-dir", default=None)

    eval_default = subparsers.add_parser("eval-default", parents=[common])
    eval_default.add_argument(
        "--default-checkpoint-dir", default=str(DEFAULT_TOP2_CHECKPOINT_DIR)
    )
    eval_default.add_argument(
        "--default-selection", choices=("best_valid", "final"), default="best_valid"
    )
    eval_default.add_argument("--wait-for-default", action="store_true")
    eval_default.add_argument("--poll-seconds", type=int, default=300)
    eval_default.add_argument("--stable-seconds", type=int, default=60)

    stats = subparsers.add_parser("stats", parents=[common])
    stats.add_argument(
        "--output", default=str(ANALYSIS_DIR / "loss_scale_statistics.csv")
    )

    run_grid_parser = subparsers.add_parser("run-grid", parents=[common])
    run_grid_parser.add_argument("--reuse-default", action="store_true", default=True)
    run_grid_parser.add_argument("--no-reuse-default", dest="reuse_default", action="store_false")
    run_grid_parser.add_argument("--wait-for-default", action="store_true", default=True)
    run_grid_parser.add_argument(
        "--no-wait-for-default", dest="wait_for_default", action="store_false"
    )
    run_grid_parser.add_argument(
        "--default-selection", choices=("best_valid", "final"), default="best_valid"
    )
    run_grid_parser.add_argument("--compute-scales", action="store_true", default=True)
    run_grid_parser.add_argument("--no-compute-scales", dest="compute_scales", action="store_false")
    run_grid_parser.add_argument("--run-pilot", action="store_true", default=True)
    run_grid_parser.add_argument("--no-run-pilot", dest="run_pilot", action="store_false")
    run_grid_parser.add_argument("--only", nargs="*", default=None)

    finalize = subparsers.add_parser("finalize-failed", parents=[common])
    finalize.add_argument("--variant-name", required=True)
    finalize.add_argument(
        "--note",
        default="Manual finalization from best validation checkpoint after failed run.",
    )

    subparsers.add_parser("implementation-check")
    subparsers.add_parser("report")
    return parser


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    if cli.command == "implementation-check":
        save_implementation_check()
    elif cli.command == "stats":
        compute_loss_scale_statistics(cli.train_data, Path(cli.output))
    elif cli.command == "train":
        train_one(cli)
    elif cli.command == "eval-default":
        eval_existing_default(cli)
    elif cli.command == "run-grid":
        run_grid(cli)
    elif cli.command == "report":
        write_report()
    elif cli.command == "finalize-failed":
        finalize_failed_cli(cli)
    else:
        parser.error(f"Unknown command: {cli.command}")


if __name__ == "__main__":
    main()
