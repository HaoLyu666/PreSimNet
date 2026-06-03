from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from tqdm import tqdm

from config import args as base_args
import model.model_MoE_gru_new as model


TYPE_LABELS = {
    0: "AV-HV",
    1: "AV-AV",
    2: "HV-HV",
    3: "HV-AV",
}
TYPE_FAMILIES = {
    0: "ACC",
    1: "ACC",
    2: "IDM",
    3: "IDM",
}
ACC_TYPES = {0, 1}
IDM_TYPES = {2, 3}
ORDERED_TYPES = [0, 1, 2, 3]

ROUTING_MODES = ["soft_all", "hard_top1", "top2", "family_top2", "oracle_top1"]
SUBSET_NAMES = [
    "all",
    "hard_top1_wrong",
    "top2_miss",
    "cross_family_top2",
    "low_confidence_max_prob_lt_0.9",
    "low_margin_top1_minus_top2_lt_0.2",
]
ACCEL_SUBSETS = [
    "all",
    "hard_top1_wrong",
    "cross_family_top2",
    "low_confidence_max_prob_lt_0.9",
    "low_margin_top1_minus_top2_lt_0.2",
]
HORIZONS = {
    "0.5s": 4,
    "1.0s": 9,
    "1.5s": 14,
    "2.0s": 19,
}
STAT_FIELDS = ["n", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max"]
ACCEL_STAT_FIELDS = [
    "n",
    "mean",
    "std",
    "min",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Routing analysis v2 for PreSimNet Mixture of Physics-encoded Experts."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkponint/"
            "ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2/"
            "epoch21_e.tar"
        ),
        help="Checkpoint path for the trained PreSimNet encoder.",
    )
    parser.add_argument(
        "--data",
        default="../data/test_data.npy",
        help="Test .npy file.",
    )
    parser.add_argument(
        "--out-dir",
        default="fig/vis/routing_analysis_v2",
        help="Output directory for CSVs, figures, metadata, and report.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="cuda:0" if t.cuda.is_available() else "cpu",
        help="Torch device for model evaluation.",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop final incomplete batch. Default keeps the full test set.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for official numbers.",
    )
    parser.add_argument(
        "--plot-sample-limit",
        type=int,
        default=250000,
        help="Maximum sample count used in distribution figures.",
    )
    parser.add_argument(
        "--accel-hist-bins",
        type=int,
        default=20000,
        help="Bins for streaming acceleration quantile summaries.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def as_float(value: float | np.floating) -> float | str:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def nanmean_or_blank(values: Iterable[float]) -> float | str:
    values_arr = np.asarray(list(values), dtype=np.float64)
    finite = values_arr[np.isfinite(values_arr)]
    if finite.size == 0:
        return ""
    return float(np.mean(finite))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def stable_absolute_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    if "?" in resolved or "\ufffd" in resolved:
        return "unavailable_due_to_wsl_non_ascii_path; use the relative path fields"
    return resolved


def stats_from_array(values: np.ndarray) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {field: (0 if field == "n" else "") for field in STAT_FIELDS}

    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
        "min": float(np.min(values)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(np.max(values)),
    }


class HistogramStats:
    """Exact mean/std/min/max plus fixed-bin quantile summaries."""

    def __init__(self, low: float, high: float, bins: int):
        self.low = float(low)
        self.high = float(high)
        self.bins = int(bins)
        self.hist = np.zeros(self.bins, dtype=np.int64)
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            return
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        values64 = values.astype(np.float64, copy=False)
        self.n += int(values64.size)
        self.sum += float(np.sum(values64, dtype=np.float64))
        self.sumsq += float(np.sum(values64 * values64, dtype=np.float64))
        self.min = min(self.min, float(np.min(values64)))
        self.max = max(self.max, float(np.max(values64)))

        clipped = np.clip(values64, self.low, self.high)
        hist, _ = np.histogram(clipped, bins=self.bins, range=(self.low, self.high))
        self.hist += hist.astype(np.int64, copy=False)

    def quantile(self, q: float) -> float:
        if self.n == 0:
            return math.nan
        threshold = max(1, int(math.ceil(q * self.n)))
        idx = int(np.searchsorted(np.cumsum(self.hist), threshold, side="left"))
        idx = min(max(idx, 0), self.bins - 1)
        width = (self.high - self.low) / self.bins
        return self.low + (idx + 0.5) * width

    def result(self) -> dict:
        if self.n == 0:
            return {field: (0 if field == "n" else "") for field in ACCEL_STAT_FIELDS}
        mean = self.sum / self.n
        var = max(self.sumsq / self.n - mean * mean, 0.0)
        return {
            "n": int(self.n),
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "min": float(self.min),
            "p05": float(self.quantile(0.05)),
            "p25": float(self.quantile(0.25)),
            "median": float(self.quantile(0.50)),
            "p75": float(self.quantile(0.75)),
            "p95": float(self.quantile(0.95)),
            "p99": float(self.quantile(0.99)),
            "max": float(self.max),
        }


class RMSEAccumulator:
    def __init__(self, out_length: int, dims: int, device: t.device):
        self.sse = t.zeros(out_length, dims, dtype=t.float64, device=device)
        self.counts = t.zeros(out_length, dims, dtype=t.float64, device=device)
        self.samples = 0

    def update(self, pred: t.Tensor, gt: t.Tensor, sample_mask: t.Tensor) -> None:
        n = int(sample_mask.sum().item())
        if n == 0:
            return
        err = (pred[sample_mask].double() - gt[sample_mask].double()).pow(2)
        self.sse += err.sum(dim=0)
        self.counts += n
        self.samples += n

    def result(self) -> np.ndarray:
        out = t.sqrt(self.sse / self.counts.clamp_min(1.0))
        out = out.detach().cpu().numpy()
        out[self.counts.detach().cpu().numpy() == 0] = np.nan
        return out


class AccelerationSummaries:
    def __init__(self, bins: int):
        self.candidate: dict[tuple[str, int], HistogramStats] = {}
        self.contribution: dict[tuple[str, str], HistogramStats] = {}
        for subset in ACCEL_SUBSETS:
            for expert_id in ORDERED_TYPES:
                self.candidate[(subset, expert_id)] = HistogramStats(-5.0, 5.0, bins)
            for quantity in [
                "abs_top2_minus_hard_top1",
                "abs_top2_minus_soft_all",
                "abs_top2_minus_family_top2",
                "abs_hard_top1_minus_oracle_top1",
            ]:
                self.contribution[(subset, quantity)] = HistogramStats(0.0, 10.0, bins)

        for quantity, bounds in [
            ("cross_family_second_expert_abs_contribution", (0.0, 5.0)),
            ("cross_family_top_expert_abs_contribution", (0.0, 5.0)),
            ("cross_family_second_contribution_ratio", (0.0, 1.0)),
        ]:
            self.contribution[("cross_family_top2", quantity)] = HistogramStats(
                bounds[0], bounds[1], bins
            )

    def update(
        self,
        matrix: np.ndarray,
        subset_masks: dict[str, np.ndarray],
    ) -> None:
        candidate_cols = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
        contribution_cols = {
            "abs_top2_minus_hard_top1": 4,
            "abs_top2_minus_soft_all": 5,
            "abs_top2_minus_family_top2": 6,
            "abs_hard_top1_minus_oracle_top1": 7,
        }
        cross_cols = {
            "cross_family_second_expert_abs_contribution": 8,
            "cross_family_top_expert_abs_contribution": 9,
            "cross_family_second_contribution_ratio": 10,
        }

        for subset in ACCEL_SUBSETS:
            mask = subset_masks[subset]
            if not np.any(mask):
                continue
            selected = matrix[mask]
            for expert_id, col in candidate_cols.items():
                self.candidate[(subset, expert_id)].update(selected[:, col])
            for quantity, col in contribution_cols.items():
                self.contribution[(subset, quantity)].update(selected[:, col])

        cross_mask = subset_masks["cross_family_top2"]
        if np.any(cross_mask):
            selected_cross = matrix[cross_mask]
            for quantity, col in cross_cols.items():
                self.contribution[("cross_family_top2", quantity)].update(
                    selected_cross[:, col]
                )

    def candidate_rows(self) -> list[dict]:
        rows = []
        for subset in ACCEL_SUBSETS:
            for expert_id in ORDERED_TYPES:
                row = {
                    "subset": subset,
                    "expert_id": expert_id,
                    "expert_label": TYPE_LABELS[expert_id],
                    "family": TYPE_FAMILIES[expert_id],
                }
                row.update(self.candidate[(subset, expert_id)].result())
                rows.append(row)
        return rows

    def contribution_rows(self) -> list[dict]:
        rows = []
        for subset, quantity in sorted(self.contribution):
            row = {"subset": subset, "quantity": quantity}
            row.update(self.contribution[(subset, quantity)].result())
            rows.append(row)
        return rows


def make_batch_iterator(
    data_path: str,
    batch_size: int,
    in_length: int,
    out_length: int,
    drop_last: bool,
    max_batches: int | None,
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = np.load(data_path, mmap_mode="r")
    total = len(data)
    usable = (total // batch_size) * batch_size if drop_last else total
    if max_batches is not None:
        usable = min(usable, max_batches * batch_size)

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
        yield start, end, hist, nextv, fut, target


def one_hot_from_indices(indices: t.Tensor, num_classes: int) -> t.Tensor:
    out = t.zeros(indices.shape[0], num_classes, device=indices.device)
    return out.scatter(1, indices.unsqueeze(1), 1.0)


def top2_route_probs(cf_probs: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    topk_vals, topk_idx = t.topk(cf_probs, k=2, dim=1)
    topk_weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-10)
    routing_weights = t.zeros_like(cf_probs)
    routing_weights.scatter_(1, topk_idx, topk_weights)
    return routing_weights, topk_idx, topk_vals, topk_weights


def family_top2_route_probs(cf_probs: t.Tensor) -> t.Tensor:
    routed = t.zeros_like(cf_probs)
    acc_mass = cf_probs[:, 0:2].sum(dim=1, keepdim=True)
    idm_mass = cf_probs[:, 2:4].sum(dim=1, keepdim=True)
    use_acc = (acc_mass >= idm_mass).squeeze(1)

    acc_weights = cf_probs[:, 0:2] / (acc_mass + 1e-10)
    idm_weights = cf_probs[:, 2:4] / (idm_mass + 1e-10)
    routed[use_acc, 0:2] = acc_weights[use_acc]
    routed[~use_acc, 2:4] = idm_weights[~use_acc]
    return routed


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


def aggregate_acceleration(route_probs: t.Tensor, all_accelerations: t.Tensor) -> t.Tensor:
    return t.bmm(route_probs.unsqueeze(1), all_accelerations).squeeze(1)


def update_state(
    veh_state: t.Tensor,
    acceleration: t.Tensor,
    next_v_t: t.Tensor,
    dt: float,
) -> t.Tensor:
    v_t = veh_state[:, 1:2]
    h_t = veh_state[:, 0:1]
    delta_v = next_v_t - v_t
    v_next = t.clamp(v_t + acceleration * dt, min=0)
    h_next = t.clamp(h_t + delta_v * dt, min=0)
    return t.cat([h_next, v_next], dim=-1)


def simulate_modes(
    simulator: model.predictor,
    params: dict,
    route_dict: dict[str, t.Tensor],
    nextv: t.Tensor,
    initial_state: t.Tensor,
    sample_masks_np: dict[str, np.ndarray],
    top_idx: t.Tensor,
    top2_norm_vals: t.Tensor,
    accel_summaries: AccelerationSummaries,
) -> dict[str, t.Tensor]:
    out_length = nextv.shape[1] - 1
    predictions: dict[str, t.Tensor] = {}

    for mode_name in ROUTING_MODES:
        veh_state = initial_state.clone()
        states = []
        route_probs = route_dict[mode_name]

        for step in range(out_length):
            next_v_t = nextv[:, step : step + 1]
            all_acc = compute_candidate_accelerations(
                simulator, params, veh_state, next_v_t
            )
            acceleration = aggregate_acceleration(route_probs, all_acc)

            if mode_name == "top2":
                update_acceleration_summaries(
                    all_acc,
                    route_dict,
                    top_idx,
                    top2_norm_vals,
                    sample_masks_np,
                    accel_summaries,
                )

            veh_state = update_state(veh_state, acceleration, next_v_t, simulator.dt)
            states.append(veh_state)

        predictions[mode_name] = t.stack(states, dim=1)

    return predictions


def update_acceleration_summaries(
    all_accelerations: t.Tensor,
    route_dict: dict[str, t.Tensor],
    top_idx: t.Tensor,
    top2_norm_vals: t.Tensor,
    sample_masks_np: dict[str, np.ndarray],
    accel_summaries: AccelerationSummaries,
) -> None:
    all_acc = all_accelerations.squeeze(-1)
    a_soft = t.sum(route_dict["soft_all"] * all_acc, dim=1)
    a_hard = t.sum(route_dict["hard_top1"] * all_acc, dim=1)
    a_top2 = t.sum(route_dict["top2"] * all_acc, dim=1)
    a_family = t.sum(route_dict["family_top2"] * all_acc, dim=1)
    a_oracle = t.sum(route_dict["oracle_top1"] * all_acc, dim=1)

    top_acc = all_acc.gather(1, top_idx[:, 0:1]).squeeze(1)
    second_acc = all_acc.gather(1, top_idx[:, 1:2]).squeeze(1)
    top_contribution = t.abs(top2_norm_vals[:, 0] * top_acc)
    second_contribution = t.abs(top2_norm_vals[:, 1] * second_acc)
    ratio = second_contribution / (top_contribution + second_contribution + 1e-10)

    matrix = t.stack(
        [
            all_acc[:, 0],
            all_acc[:, 1],
            all_acc[:, 2],
            all_acc[:, 3],
            t.abs(a_top2 - a_hard),
            t.abs(a_top2 - a_soft),
            t.abs(a_top2 - a_family),
            t.abs(a_hard - a_oracle),
            second_contribution,
            top_contribution,
            ratio,
        ],
        dim=1,
    )
    accel_summaries.update(matrix.detach().cpu().numpy(), sample_masks_np)


def rmse_row(
    mode: str,
    subset: str,
    accumulator: RMSEAccumulator,
    checkpoint: str,
) -> dict:
    values = accumulator.result()
    gap = values[:, 0]
    vel = values[:, 1]
    row = {
        "routing_mode": mode,
        "subset": subset,
        "sample_count": accumulator.samples,
        "checkpoint_path": checkpoint,
    }
    for label, idx in HORIZONS.items():
        row[f"gap_rmse_{label}"] = as_float(gap[idx])
    row["gap_avg_4h"] = nanmean_or_blank([gap[idx] for idx in HORIZONS.values()])
    row["gap_avg_20step"] = nanmean_or_blank(gap)
    for label, idx in HORIZONS.items():
        row[f"velocity_rmse_{label}"] = as_float(vel[idx])
    row["velocity_avg_4h"] = nanmean_or_blank([vel[idx] for idx in HORIZONS.values()])
    row["velocity_avg_20step"] = nanmean_or_blank(vel)
    return row


def official_top2_rows(top2_acc: RMSEAccumulator, checkpoint: str) -> list[dict]:
    values = top2_acc.result()
    rows = []
    for metric_name, metric_values in [
        ("gap_rmse", values[:, 0]),
        ("velocity_rmse", values[:, 1]),
    ]:
        row = {
            "metric_name": metric_name,
            "routing_mode": "top2",
            "sample_count": top2_acc.samples,
            "checkpoint_path": checkpoint,
        }
        for label, idx in HORIZONS.items():
            row[label] = as_float(metric_values[idx])
        row["avg_4h"] = nanmean_or_blank(
            [metric_values[idx] for idx in HORIZONS.values()]
        )
        row["avg_20step"] = nanmean_or_blank(metric_values)
        rows.append(row)
    return rows


def build_weight_rows(weight_data: dict, subsets: dict[str, np.ndarray]) -> list[dict]:
    metrics = {
        "top1_raw_softmax_probability": weight_data["top1_raw"],
        "top2_raw_softmax_probability": weight_data["top2_raw"],
        "top1_top2_renormalized_weight": weight_data["top1_renorm"],
        "top2_top2_renormalized_weight": weight_data["top2_renorm"],
        "ground_truth_expert_probability": weight_data["true_prob"],
        "same_family_mass_as_top1": weight_data["same_family_mass"],
        "opposite_family_mass_as_top1": weight_data["opposite_family_mass"],
    }
    rows = []
    for subset_name in SUBSET_NAMES:
        mask = subsets[subset_name]
        for metric_name, values in metrics.items():
            row = {"subset": subset_name, "metric": metric_name}
            row.update(stats_from_array(values[mask]))
            rows.append(row)
    return rows


def build_grouped_weight_rows(
    weight_data: dict,
    masks: dict[str, np.ndarray],
    group_names: list[str],
) -> list[dict]:
    metrics = {
        "top1_raw_softmax_probability": weight_data["top1_raw"],
        "top2_raw_softmax_probability": weight_data["top2_raw"],
        "top1_top2_renormalized_weight": weight_data["top1_renorm"],
        "top2_top2_renormalized_weight": weight_data["top2_renorm"],
        "ground_truth_expert_probability": weight_data["true_prob"],
        "same_family_mass_as_top1": weight_data["same_family_mass"],
        "opposite_family_mass_as_top1": weight_data["opposite_family_mass"],
    }
    rows = []
    for group in group_names:
        mask = masks[group]
        for metric_name, values in metrics.items():
            row = {"group": group, "metric": metric_name}
            row.update(stats_from_array(values[mask]))
            rows.append(row)
    return rows


def build_selection_rows(
    total_by_type: np.ndarray,
    hard_wrong_by_type: np.ndarray,
    top2_miss_by_type: np.ndarray,
    top2_rescue_by_type: np.ndarray,
    cross_family_by_type: np.ndarray,
    low_conf_by_type: np.ndarray,
    low_margin_by_type: np.ndarray,
    scope: str,
) -> list[dict]:
    rows = []
    if scope == "overall":
        total = int(total_by_type.sum())
        hard_wrong = int(hard_wrong_by_type.sum())
        top2_miss = int(top2_miss_by_type.sum())
        rescue = int(top2_rescue_by_type.sum())
        cross_family = int(cross_family_by_type.sum())
        low_conf = int(low_conf_by_type.sum())
        low_margin = int(low_margin_by_type.sum())
        rows.append(
            {
                "scope": "overall",
                "type_id": "all",
                "type_label": "ALL",
                "family": "ALL",
                "total_samples": total,
                "hard_top1_wrong_count": hard_wrong,
                "hard_top1_wrong_rate": safe_div(hard_wrong, total),
                "top2_miss_count": top2_miss,
                "top2_miss_rate": safe_div(top2_miss, total),
                "top2_rescue_count": rescue,
                "top2_rescue_rate_among_hard_wrong": safe_div(rescue, hard_wrong),
                "cross_family_top2_count": cross_family,
                "cross_family_top2_rate": safe_div(cross_family, total),
                "low_confidence_count": low_conf,
                "low_confidence_rate": safe_div(low_conf, total),
                "low_margin_count": low_margin,
                "low_margin_rate": safe_div(low_margin, total),
            }
        )
        return rows

    for type_id in ORDERED_TYPES:
        total = int(total_by_type[type_id])
        hard_wrong = int(hard_wrong_by_type[type_id])
        top2_miss = int(top2_miss_by_type[type_id])
        rescue = int(top2_rescue_by_type[type_id])
        cross_family = int(cross_family_by_type[type_id])
        low_conf = int(low_conf_by_type[type_id])
        low_margin = int(low_margin_by_type[type_id])
        rows.append(
            {
                "scope": "by_type",
                "type_id": type_id,
                "type_label": TYPE_LABELS[type_id],
                "family": TYPE_FAMILIES[type_id],
                "total_samples": total,
                "hard_top1_wrong_count": hard_wrong,
                "hard_top1_wrong_rate": safe_div(hard_wrong, total),
                "top2_miss_count": top2_miss,
                "top2_miss_rate": safe_div(top2_miss, total),
                "top2_rescue_count": rescue,
                "top2_rescue_rate_among_hard_wrong": safe_div(rescue, hard_wrong),
                "cross_family_top2_count": cross_family,
                "cross_family_top2_rate": safe_div(cross_family, total),
                "low_confidence_count": low_conf,
                "low_confidence_rate": safe_div(low_conf, total),
                "low_margin_count": low_margin,
                "low_margin_rate": safe_div(low_margin, total),
            }
        )
    return rows


def plot_selection(out_dir: Path, by_type_rows: list[dict]) -> None:
    labels = [row["type_label"] for row in by_type_rows]
    hard = np.array([row["hard_top1_wrong_rate"] for row in by_type_rows]) * 100.0
    miss = np.array([row["top2_miss_rate"] for row in by_type_rows]) * 100.0
    rescue = (
        np.array([row["top2_rescue_rate_among_hard_wrong"] for row in by_type_rows])
        * 100.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, hard, width, label="Hard Top-1 wrong")
    axes[0].bar(x + width / 2, miss, width, label="Top-2 miss")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Selection error rates")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x, rescue, color="#2c7fb8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Rescue rate among hard errors (%)")
    axes[1].set_ylim(0, max(100.0, float(np.nanmax(rescue)) * 1.08))
    axes[1].set_title("Top-2 rescue rate")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "selection_error_and_rescue_rates.png", bbox_inches="tight")
    plt.close(fig)


def sample_for_plot(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= limit:
        return values
    idx = rng.choice(values.size, size=limit, replace=False)
    return values[idx]


def plot_weight_distributions(
    out_dir: Path,
    weight_data: dict,
    masks: dict[str, np.ndarray],
    sample_limit: int,
) -> None:
    rng = np.random.default_rng(72)
    panels = [
        ("all", "All samples"),
        ("hard_top1_wrong", "Hard Top-1 wrong"),
    ]
    metrics = [
        ("top1_raw", "Top-1 raw"),
        ("top2_raw", "Top-2 raw"),
        ("top1_renorm", "Top-1 renorm"),
        ("top2_renorm", "Top-2 renorm"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    for ax, (subset, title) in zip(axes, panels):
        mask = masks[subset]
        data = [
            sample_for_plot(weight_data[key][mask], sample_limit, rng) for key, _ in metrics
        ]
        nonempty_positions = [
            i + 1 for i, values in enumerate(data) if np.asarray(values).size > 0
        ]
        nonempty_data = [values for values in data if np.asarray(values).size > 0]
        if nonempty_data:
            ax.violinplot(
                nonempty_data,
                positions=nonempty_positions,
                showmeans=True,
                showextrema=False,
            )
            ax.boxplot(
                nonempty_data,
                positions=nonempty_positions,
                widths=0.18,
                showfliers=False,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xticks(np.arange(1, len(metrics) + 1))
        ax.set_xticklabels([label for _, label in metrics], rotation=18, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Probability / renormalized weight")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "weight_distribution_top1_top2.png", bbox_inches="tight")
    plt.close(fig)

    cross = masks["cross_family_top2"]
    if np.any(cross):
        fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
        vals = sample_for_plot(weight_data["top2_renorm"][cross], sample_limit, rng)
        ax.hist(vals, bins=80, color="#4f8fc9", alpha=0.85)
        ax.set_xlabel("Second expert Top-2-renormalized weight")
        ax.set_ylabel("Samples")
        ax.set_title("Cross-family second-expert weight")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "cross_family_second_weight_distribution.png", bbox_inches="tight")
        plt.close(fig)


def plot_acceleration_ratio(out_dir: Path, accel_summaries: AccelerationSummaries) -> None:
    key = ("cross_family_top2", "cross_family_second_contribution_ratio")
    hist_stats = accel_summaries.contribution.get(key)
    if hist_stats is None or hist_stats.n == 0:
        return
    edges = np.linspace(hist_stats.low, hist_stats.high, hist_stats.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    ax.plot(centers, hist_stats.hist, color="#2c7fb8", linewidth=1.4)
    ax.set_xlabel("Second-expert contribution ratio")
    ax.set_ylabel("Sample-steps")
    ax.set_title("Cross-family acceleration contribution ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_family_acceleration_contribution_ratio.png", bbox_inches="tight")
    plt.close(fig)


def make_implementation_check(metadata: dict) -> str:
    mapping_lines = "\n".join(
        [
            "| Type id | Label | Family | Expert model |",
            "|---:|---|---|---|",
        ]
        + [
            f"| {type_id} | {TYPE_LABELS[type_id]} | {TYPE_FAMILIES[type_id]} | "
            f"{'ACC' if type_id in ACC_TYPES else 'IDM'} |"
            for type_id in ORDERED_TYPES
        ]
    )
    return f"""# Routing Implementation Check

This file records the implementation facts verified for the v2 routing analysis. It is an analysis note only; no model architecture was changed and no training was performed.

## Verified Label Mapping

{mapping_lines}

Evidence:

- `loader2.py` reads the ground-truth interaction id from `window[0, 3]` and one-hot encodes it as `cf_type[int(window[0, 3])] = 1`.
- `visualize_trajectories.py` and the existing routing analysis use the semantic mapping above.
- The model code assigns types 0/1 to the ACC family and types 2/3 to the IDM family in `predictor.veh_dynamic`.

## Logits And Probabilities

- `Encoder.forward` predicts `cf_type_pred` from the history/type query branch.
- `Encoder.forward` computes `cf_type_probs = F.softmax(cf_type_pred, dim=-1)` for the trajectory-memory mapping.
- `ParameterPredictionHead.forward` also computes `cf_probs = F.softmax(cf_type_pred, dim=-1)` and returns it in `model_outputs["params"]["cf_probs"]`.
- The v2 script routes from these model-produced softmax probabilities. Ground-truth labels are not used for `soft_all`, `hard_top1`, `top2`, or `family_top2`.

## Expert Acceleration Computation

- `ParameterPredictionHead` creates four independent expert parameter heads.
- The first two parameter heads are returned as `acc_params`; the last two are returned as `idm_params`.
- `predictor.veh_dynamic` computes all four candidate accelerations first: types 0/1 through `acc_model`, and types 2/3 through `idm_model`.
- The route weights are applied only after candidate accelerations exist, through a weighted acceleration sum. The v2 script follows the same acceleration-level aggregation and never aggregates ACC/IDM parameters.

## Current Predictor Default

- `predictor.route_probabilities` defaults to `aggregation_mode="soft_all"` when no aggregation mode is supplied.
- The model file also has helper branches for `top2`/`topk` and `hard_top1`, but normal evaluation with the existing config follows full soft-all acceleration aggregation unless the caller overrides the route weights.

## v2 Routing Modes

- `soft_all`: all four candidate accelerations are aggregated with raw `cf_probs`.
- `hard_top1`: only `argmax(cf_probs)` is selected.
- `top2`: the two largest raw probabilities are selected, renormalized, and used to aggregate candidate accelerations.
- `family_top2`: ACC mass `p0+p1` is compared with IDM mass `p2+p3`; only the selected family is renormalized and aggregated.
- `oracle_top1`: diagnostic only; selects the ground-truth expert.

## Average Definitions

- `avg_4h`: mean of the four reported horizons: 0.5s, 1.0s, 1.5s, and 2.0s.
- `avg_20step`: mean of the per-step RMSE over all 20 prediction steps.
- RMSE aggregation in v2: global squared-error accumulation over the selected samples, followed by square root.

## Run Metadata

- Data: `{metadata["data_path"]}`
- Checkpoint: `{metadata["checkpoint_path"]}`
- Output directory: `{metadata["output_dir"]}`
- Sample count: `{metadata["sample_count"]}`
- Device: `{metadata["device"]}`
- Rerun command: `{metadata["rerun_command"]}`
"""


def format_float(value: object, digits: int = 6) -> str:
    if value == "" or value is None:
        return ""
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value_float:.{digits}f}"


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 6) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, (float, np.floating)):
                cells.append(format_float(value, digits))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def make_report(
    metadata: dict,
    official_rows: list[dict],
    summary_rows: list[dict],
    by_type_rows: list[dict],
    rmse_rows: list[dict],
    weight_rows: list[dict],
    accel_rows: list[dict],
) -> str:
    overall = summary_rows[0]
    top1_wrong_rows = [
        row for row in rmse_rows if row["subset"] == "hard_top1_wrong"
    ]
    low_rows = [
        row
        for row in rmse_rows
        if row["subset"]
        in {"low_confidence_max_prob_lt_0.9", "low_margin_top1_minus_top2_lt_0.2"}
        and row["routing_mode"] in {"hard_top1", "top2", "soft_all", "family_top2"}
    ]
    cross_weight_key_rows = [
        row
        for row in weight_rows
        if row["subset"] == "cross_family_top2"
        and row["metric"]
        in {
            "top1_raw_softmax_probability",
            "top2_raw_softmax_probability",
            "top2_top2_renormalized_weight",
            "same_family_mass_as_top1",
            "opposite_family_mass_as_top1",
        }
    ]
    accel_key_rows = [
        row
        for row in accel_rows
        if row["subset"] in {"all", "hard_top1_wrong", "cross_family_top2"}
        and row["quantity"]
        in {
            "abs_top2_minus_hard_top1",
            "abs_top2_minus_soft_all",
            "abs_top2_minus_family_top2",
            "cross_family_second_contribution_ratio",
        }
    ]

    facts = [
        (
            f"Hard Top-1 wrong: {overall['hard_top1_wrong_count']} / "
            f"{overall['total_samples']} = "
            f"{overall['hard_top1_wrong_rate'] * 100:.4f}%."
        ),
        (
            f"Top-2 miss: {overall['top2_miss_count']} / "
            f"{overall['total_samples']} = {overall['top2_miss_rate'] * 100:.4f}%."
        ),
        (
            f"Top-2 rescue among hard Top-1 wrong samples: "
            f"{overall['top2_rescue_count']} / {overall['hard_top1_wrong_count']} = "
            f"{overall['top2_rescue_rate_among_hard_wrong'] * 100:.2f}%."
        ),
        (
            f"Cross-family Top-2 pairs: {overall['cross_family_top2_count']} / "
            f"{overall['total_samples']} = "
            f"{overall['cross_family_top2_rate'] * 100:.2f}%."
        ),
    ]

    hard_all = next(
        row
        for row in rmse_rows
        if row["routing_mode"] == "hard_top1" and row["subset"] == "all"
    )
    top2_all = next(
        row for row in rmse_rows if row["routing_mode"] == "top2" and row["subset"] == "all"
    )
    if (
        hard_all["gap_avg_4h"] != ""
        and top2_all["gap_avg_4h"] != ""
        and float(hard_all["gap_avg_4h"]) < float(top2_all["gap_avg_4h"])
    ):
        facts.append(
            "Unfavorable/neutral result to keep visible: hard Top-1 has lower overall "
            f"gap avg_4h ({float(hard_all['gap_avg_4h']):.6f}) than Top-2 "
            f"({float(top2_all['gap_avg_4h']):.6f}), while Top-2 is mainly evaluated "
            "as a buffer on ambiguous/misclassified subsets."
        )

    return f"""# Routing Analysis v2

This report summarizes computed routing-analysis materials for the PreSimNet Mixture of Physics-encoded Experts. It is not a reviewer response, not manuscript text, and not LaTeX.

## Implementation Check

- Verified mapping: type 0 = AV-HV/ACC, type 1 = AV-AV/ACC, type 2 = HV-HV/IDM, type 3 = HV-AV/IDM.
- `cf_probs` are model softmax probabilities from `cf_type_pred`; normal routing modes do not use ground-truth labels.
- Top-2 and all other non-oracle modes aggregate candidate accelerations, not ACC/IDM parameters.
- `oracle_top1` uses ground-truth labels only as a diagnostic upper-bound route.
- Acceleration contribution diagnostics use the official Top-2 rollout state at each step as the common state for comparing route-weighted accelerations.

## Paths And Command

- Data: `{metadata["data_path"]}`
- Checkpoint: `{metadata["checkpoint_path"]}`
- Output directory: `{metadata["output_dir"]}`
- Samples evaluated: `{metadata["sample_count"]}`
- Rerun: `{metadata["rerun_command"]}`

## Average Definitions

- `avg_4h`: mean of RMSE at 0.5s, 1.0s, 1.5s, and 2.0s.
- `avg_20step`: mean of per-step RMSE over all 20 prediction steps.
- RMSE is globally accumulated over selected samples before taking the square root.

## Official Top-2 Evaluation

{markdown_table(official_rows, [
    ("Metric", "metric_name"),
    ("0.5s", "0.5s"),
    ("1.0s", "1.0s"),
    ("1.5s", "1.5s"),
    ("2.0s", "2.0s"),
    ("avg_4h", "avg_4h"),
    ("avg_20step", "avg_20step"),
    ("n", "sample_count"),
], digits=6)}

## Selection Statistics

{markdown_table(summary_rows, [
    ("Scope", "scope"),
    ("n", "total_samples"),
    ("Hard wrong", "hard_top1_wrong_count"),
    ("Hard wrong rate", "hard_top1_wrong_rate"),
    ("Top-2 miss", "top2_miss_count"),
    ("Top-2 miss rate", "top2_miss_rate"),
    ("Rescue", "top2_rescue_count"),
    ("Rescue rate", "top2_rescue_rate_among_hard_wrong"),
    ("Cross-family", "cross_family_top2_count"),
    ("Cross-family rate", "cross_family_top2_rate"),
], digits=6)}

By type:

{markdown_table(by_type_rows, [
    ("Type", "type_label"),
    ("Family", "family"),
    ("n", "total_samples"),
    ("Hard wrong rate", "hard_top1_wrong_rate"),
    ("Top-2 miss rate", "top2_miss_rate"),
    ("Rescue rate", "top2_rescue_rate_among_hard_wrong"),
    ("Cross-family rate", "cross_family_top2_rate"),
], digits=6)}

## RMSE On Hard Top-1 Wrong Samples

{markdown_table(top1_wrong_rows, [
    ("Mode", "routing_mode"),
    ("n", "sample_count"),
    ("Gap 0.5s", "gap_rmse_0.5s"),
    ("Gap 1.0s", "gap_rmse_1.0s"),
    ("Gap 1.5s", "gap_rmse_1.5s"),
    ("Gap 2.0s", "gap_rmse_2.0s"),
    ("Gap avg_4h", "gap_avg_4h"),
    ("Gap avg_20step", "gap_avg_20step"),
    ("Vel avg_4h", "velocity_avg_4h"),
    ("Vel avg_20step", "velocity_avg_20step"),
], digits=6)}

## Low-Confidence And Low-Margin Subsets

{markdown_table(low_rows, [
    ("Subset", "subset"),
    ("Mode", "routing_mode"),
    ("n", "sample_count"),
    ("Gap avg_4h", "gap_avg_4h"),
    ("Gap avg_20step", "gap_avg_20step"),
    ("Vel avg_4h", "velocity_avg_4h"),
    ("Vel avg_20step", "velocity_avg_20step"),
], digits=6)}

## Cross-Family Top-2 Weight Analysis

{markdown_table(cross_weight_key_rows, [
    ("Metric", "metric"),
    ("n", "n"),
    ("mean", "mean"),
    ("std", "std"),
    ("p05", "p05"),
    ("median", "median"),
    ("p95", "p95"),
    ("max", "max"),
], digits=6)}

## Acceleration-Level Contribution Analysis

{markdown_table(accel_key_rows, [
    ("Subset", "subset"),
    ("Quantity", "quantity"),
    ("n", "n"),
    ("mean", "mean"),
    ("std", "std"),
    ("p05", "p05"),
    ("median", "median"),
    ("p95", "p95"),
    ("p99", "p99"),
    ("max", "max"),
], digits=6)}

Acceleration quantiles in this table are fixed-bin streaming summaries; means, standard deviations, minima, and maxima are exact over the streamed finite values.

## Strongest Quantitative Facts

""" + "\n".join(f"- {fact}" for fact in facts) + """

## Quality Checks

- No retraining was performed.
- Ground-truth labels were not used for normal `soft_all`, `hard_top1`, `top2`, or `family_top2` inference.
- `oracle_top1` is marked diagnostic only.
- Top-2 aggregates candidate accelerations, not parameters.
- `avg_4h` and `avg_20step` are reported separately.
- Sample counts are included in each CSV and table.
- The report includes favorable and unfavorable/neutral results where present.
"""


def main() -> None:
    cli = parse_args()
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    device = t.device(cli.device)
    if device.type == "cuda":
        t.set_float32_matmul_precision("high")

    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False
    run_args["batch_size"] = cli.batch_size

    encoder = model.Encoder(run_args).to(device)
    checkpoint = t.load(cli.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    simulator = model.predictor(run_args)

    data = np.load(cli.data, mmap_mode="r")
    total_data_samples = int(len(data))
    del data
    usable_samples = (
        (total_data_samples // cli.batch_size) * cli.batch_size
        if cli.drop_last
        else total_data_samples
    )
    if cli.max_batches is not None:
        usable_samples = min(usable_samples, cli.max_batches * cli.batch_size)
    total_batches = (usable_samples + cli.batch_size - 1) // cli.batch_size

    out_length = int(run_args["out_length"])
    cf_type = int(run_args["cf_type"])

    total_by_type = np.zeros(cf_type, dtype=np.int64)
    hard_wrong_by_type = np.zeros(cf_type, dtype=np.int64)
    top2_miss_by_type = np.zeros(cf_type, dtype=np.int64)
    top2_rescue_by_type = np.zeros(cf_type, dtype=np.int64)
    cross_family_by_type = np.zeros(cf_type, dtype=np.int64)
    low_conf_by_type = np.zeros(cf_type, dtype=np.int64)
    low_margin_by_type = np.zeros(cf_type, dtype=np.int64)
    confusion = np.zeros((cf_type, cf_type), dtype=np.int64)
    top2_pair_counts = np.zeros((cf_type, cf_type), dtype=np.int64)

    rmse_accumulators = {
        mode_name: {
            subset_name: RMSEAccumulator(out_length, 2, device)
            for subset_name in SUBSET_NAMES
        }
        for mode_name in ROUTING_MODES
    }
    accel_summaries = AccelerationSummaries(cli.accel_hist_bins)

    weight_chunks: dict[str, list[np.ndarray]] = {
        "target": [],
        "top1_idx": [],
        "top2_idx": [],
        "top1_raw": [],
        "top2_raw": [],
        "top1_renorm": [],
        "top2_renorm": [],
        "true_prob": [],
        "same_family_mass": [],
        "opposite_family_mass": [],
        "hard_top1_correct": [],
        "top2_hit": [],
        "cross_family_top2": [],
        "low_confidence": [],
        "low_margin": [],
    }

    iterator = make_batch_iterator(
        cli.data,
        cli.batch_size,
        run_args["in_length"],
        run_args["out_length"],
        cli.drop_last,
        cli.max_batches,
    )

    with t.inference_mode():
        for _, _, hist_np, nextv_np, fut_np, target_np in tqdm(
            iterator, total=total_batches, desc="Routing analysis v2"
        ):
            hist = t.as_tensor(hist_np, device=device)
            nextv = t.as_tensor(nextv_np, device=device)
            fut = t.as_tensor(fut_np, device=device)
            target = t.as_tensor(target_np, dtype=t.long, device=device)

            model_outputs = encoder(hist)
            params = model_outputs["params"]
            cf_probs = params["cf_probs"]

            top2_route, top_idx, top_vals, top2_norm_vals = top2_route_probs(cf_probs)
            top1_idx = top_idx[:, 0]
            hard_route = one_hot_from_indices(top1_idx, cf_type)
            soft_route = cf_probs
            family_route = family_top2_route_probs(cf_probs)
            oracle_route = one_hot_from_indices(target, cf_type)

            route_dict = {
                "soft_all": soft_route,
                "hard_top1": hard_route,
                "top2": top2_route,
                "family_top2": family_route,
                "oracle_top1": oracle_route,
            }

            top2_hit_t = (top_idx[:, 0] == target) | (top_idx[:, 1] == target)
            hard_wrong_t = top1_idx != target
            top2_miss_t = ~top2_hit_t
            cross_family_t = (top_idx[:, 0] < 2) != (top_idx[:, 1] < 2)
            low_conf_t = top_vals[:, 0] < 0.9
            low_margin_t = (top_vals[:, 0] - top_vals[:, 1]) < 0.2

            subset_masks_t = {
                "all": t.ones_like(target, dtype=t.bool),
                "hard_top1_wrong": hard_wrong_t,
                "top2_miss": top2_miss_t,
                "cross_family_top2": cross_family_t,
                "low_confidence_max_prob_lt_0.9": low_conf_t,
                "low_margin_top1_minus_top2_lt_0.2": low_margin_t,
            }
            sample_masks_np = {
                key: value.detach().cpu().numpy().astype(bool, copy=False)
                for key, value in subset_masks_t.items()
            }

            target_cpu = target_np.astype(np.int64, copy=False)
            top1_cpu = top1_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            top2_cpu = top_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            top_vals_cpu = top_vals.detach().cpu().numpy().astype(np.float32, copy=False)
            norm_vals_cpu = (
                top2_norm_vals.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            probs_cpu = cf_probs.detach().cpu().numpy().astype(np.float32, copy=False)

            top1_correct_cpu = top1_cpu == target_cpu
            top2_hit_cpu = sample_masks_np["all"] & ~sample_masks_np["top2_miss"]
            cross_cpu = sample_masks_np["cross_family_top2"]
            low_conf_cpu = sample_masks_np["low_confidence_max_prob_lt_0.9"]
            low_margin_cpu = sample_masks_np["low_margin_top1_minus_top2_lt_0.2"]
            rescue_cpu = (~top1_correct_cpu) & top2_hit_cpu

            top1_is_acc_cpu = top1_cpu < 2
            same_family_mass = np.where(
                top1_is_acc_cpu,
                probs_cpu[:, 0] + probs_cpu[:, 1],
                probs_cpu[:, 2] + probs_cpu[:, 3],
            ).astype(np.float32, copy=False)
            opposite_family_mass = (1.0 - same_family_mass).astype(np.float32, copy=False)
            true_prob = probs_cpu[np.arange(len(target_cpu)), target_cpu].astype(
                np.float32, copy=False
            )

            for type_id in ORDERED_TYPES:
                type_mask = target_cpu == type_id
                total_by_type[type_id] += int(type_mask.sum())
                hard_wrong_by_type[type_id] += int((type_mask & ~top1_correct_cpu).sum())
                top2_miss_by_type[type_id] += int(
                    (type_mask & ~top2_hit_cpu).sum()
                )
                top2_rescue_by_type[type_id] += int((type_mask & rescue_cpu).sum())
                cross_family_by_type[type_id] += int((type_mask & cross_cpu).sum())
                low_conf_by_type[type_id] += int((type_mask & low_conf_cpu).sum())
                low_margin_by_type[type_id] += int((type_mask & low_margin_cpu).sum())

            np.add.at(confusion, (target_cpu, top1_cpu), 1)
            np.add.at(top2_pair_counts, (top2_cpu[:, 0], top2_cpu[:, 1]), 1)

            weight_chunks["target"].append(target_cpu.astype(np.int16, copy=False))
            weight_chunks["top1_idx"].append(top1_cpu.astype(np.int16, copy=False))
            weight_chunks["top2_idx"].append(top2_cpu[:, 1].astype(np.int16, copy=False))
            weight_chunks["top1_raw"].append(top_vals_cpu[:, 0])
            weight_chunks["top2_raw"].append(top_vals_cpu[:, 1])
            weight_chunks["top1_renorm"].append(norm_vals_cpu[:, 0])
            weight_chunks["top2_renorm"].append(norm_vals_cpu[:, 1])
            weight_chunks["true_prob"].append(true_prob)
            weight_chunks["same_family_mass"].append(same_family_mass)
            weight_chunks["opposite_family_mass"].append(opposite_family_mass)
            weight_chunks["hard_top1_correct"].append(top1_correct_cpu)
            weight_chunks["top2_hit"].append(top2_hit_cpu)
            weight_chunks["cross_family_top2"].append(cross_cpu)
            weight_chunks["low_confidence"].append(low_conf_cpu)
            weight_chunks["low_margin"].append(low_margin_cpu)

            initial_state = hist[:, -1, [4, 2]]
            predictions = simulate_modes(
                simulator,
                params,
                route_dict,
                nextv,
                initial_state,
                sample_masks_np,
                top_idx,
                top2_norm_vals,
                accel_summaries,
            )

            gt_dynamic = fut[:, :, :2]
            for mode_name, pred in predictions.items():
                for subset_name, subset_mask in subset_masks_t.items():
                    rmse_accumulators[mode_name][subset_name].update(
                        pred, gt_dynamic, subset_mask
                    )

    weight_data = {
        key: np.concatenate(chunks, axis=0) for key, chunks in weight_chunks.items()
    }
    subset_masks = {
        "all": np.ones_like(weight_data["target"], dtype=bool),
        "hard_top1_wrong": ~weight_data["hard_top1_correct"].astype(bool),
        "top2_miss": ~weight_data["top2_hit"].astype(bool),
        "cross_family_top2": weight_data["cross_family_top2"].astype(bool),
        "low_confidence_max_prob_lt_0.9": weight_data["low_confidence"].astype(bool),
        "low_margin_top1_minus_top2_lt_0.2": weight_data["low_margin"].astype(bool),
    }

    selection_fields = [
        "scope",
        "type_id",
        "type_label",
        "family",
        "total_samples",
        "hard_top1_wrong_count",
        "hard_top1_wrong_rate",
        "top2_miss_count",
        "top2_miss_rate",
        "top2_rescue_count",
        "top2_rescue_rate_among_hard_wrong",
        "cross_family_top2_count",
        "cross_family_top2_rate",
        "low_confidence_count",
        "low_confidence_rate",
        "low_margin_count",
        "low_margin_rate",
    ]
    summary_rows = build_selection_rows(
        total_by_type,
        hard_wrong_by_type,
        top2_miss_by_type,
        top2_rescue_by_type,
        cross_family_by_type,
        low_conf_by_type,
        low_margin_by_type,
        "overall",
    )
    by_type_rows = build_selection_rows(
        total_by_type,
        hard_wrong_by_type,
        top2_miss_by_type,
        top2_rescue_by_type,
        cross_family_by_type,
        low_conf_by_type,
        low_margin_by_type,
        "by_type",
    )
    write_csv(out_dir / "routing_selection_summary.csv", summary_rows, selection_fields)
    write_csv(out_dir / "routing_selection_by_type.csv", by_type_rows, selection_fields)

    confusion_rows = []
    for true_type in ORDERED_TYPES:
        row = {
            "true_type_id": true_type,
            "true_type_label": TYPE_LABELS[true_type],
            "family": TYPE_FAMILIES[true_type],
        }
        for pred_type in ORDERED_TYPES:
            row[f"pred_type_{pred_type}_{TYPE_LABELS[pred_type]}"] = int(
                confusion[true_type, pred_type]
            )
        confusion_rows.append(row)
    write_csv(
        out_dir / "hard_top1_confusion_matrix.csv",
        confusion_rows,
        ["true_type_id", "true_type_label", "family"]
        + [f"pred_type_{i}_{TYPE_LABELS[i]}" for i in ORDERED_TYPES],
    )

    pair_rows = []
    for first in ORDERED_TYPES:
        for second in ORDERED_TYPES:
            count = int(top2_pair_counts[first, second])
            pair_rows.append(
                {
                    "top1_type_id": first,
                    "top1_label": TYPE_LABELS[first],
                    "top1_family": TYPE_FAMILIES[first],
                    "second_type_id": second,
                    "second_label": TYPE_LABELS[second],
                    "second_family": TYPE_FAMILIES[second],
                    "count": count,
                    "rate": safe_div(count, usable_samples),
                    "cross_family": int(TYPE_FAMILIES[first] != TYPE_FAMILIES[second]),
                }
            )
    write_csv(
        out_dir / "top2_pair_counts.csv",
        pair_rows,
        [
            "top1_type_id",
            "top1_label",
            "top1_family",
            "second_type_id",
            "second_label",
            "second_family",
            "count",
            "rate",
            "cross_family",
        ],
    )

    rmse_rows = []
    for mode_name in ROUTING_MODES:
        for subset_name in SUBSET_NAMES:
            rmse_rows.append(
                rmse_row(
                    mode_name,
                    subset_name,
                    rmse_accumulators[mode_name][subset_name],
                    cli.checkpoint,
                )
            )
    rmse_fields = [
        "routing_mode",
        "subset",
        "sample_count",
        "checkpoint_path",
        "gap_rmse_0.5s",
        "gap_rmse_1.0s",
        "gap_rmse_1.5s",
        "gap_rmse_2.0s",
        "gap_avg_4h",
        "gap_avg_20step",
        "velocity_rmse_0.5s",
        "velocity_rmse_1.0s",
        "velocity_rmse_1.5s",
        "velocity_rmse_2.0s",
        "velocity_avg_4h",
        "velocity_avg_20step",
    ]
    write_csv(out_dir / "routing_rmse_by_mode_and_subset.csv", rmse_rows, rmse_fields)

    official_rows = official_top2_rows(
        rmse_accumulators["top2"]["all"], cli.checkpoint
    )
    official_fields = [
        "metric_name",
        "0.5s",
        "1.0s",
        "1.5s",
        "2.0s",
        "avg_4h",
        "avg_20step",
        "sample_count",
        "checkpoint_path",
        "routing_mode",
    ]
    write_csv(out_dir / "official_top2_eval.csv", official_rows, official_fields)

    weight_rows = build_weight_rows(weight_data, subset_masks)
    write_csv(
        out_dir / "routing_weight_summary.csv",
        weight_rows,
        ["subset", "metric"] + STAT_FIELDS,
    )

    cross_masks = {"cross_family_top2": subset_masks["cross_family_top2"]}
    for first in ORDERED_TYPES:
        for second in ORDERED_TYPES:
            if TYPE_FAMILIES[first] == TYPE_FAMILIES[second]:
                continue
            key = f"top1_{first}_{TYPE_LABELS[first]}__second_{second}_{TYPE_LABELS[second]}"
            cross_masks[key] = (
                (weight_data["top1_idx"] == first)
                & (weight_data["top2_idx"] == second)
            )
    cross_rows = build_grouped_weight_rows(
        weight_data, cross_masks, list(cross_masks.keys())
    )
    write_csv(
        out_dir / "cross_family_weight_summary.csv",
        cross_rows,
        ["group", "metric"] + STAT_FIELDS,
    )

    low_rows = build_grouped_weight_rows(
        weight_data,
        {
            "low_confidence_max_prob_lt_0.9": subset_masks[
                "low_confidence_max_prob_lt_0.9"
            ],
            "low_margin_top1_minus_top2_lt_0.2": subset_masks[
                "low_margin_top1_minus_top2_lt_0.2"
            ],
        },
        [
            "low_confidence_max_prob_lt_0.9",
            "low_margin_top1_minus_top2_lt_0.2",
        ],
    )
    write_csv(
        out_dir / "low_confidence_weight_summary.csv",
        low_rows,
        ["group", "metric"] + STAT_FIELDS,
    )

    candidate_rows = accel_summaries.candidate_rows()
    contribution_rows = accel_summaries.contribution_rows()
    write_csv(
        out_dir / "candidate_acceleration_stats.csv",
        candidate_rows,
        ["subset", "expert_id", "expert_label", "family"] + ACCEL_STAT_FIELDS,
    )
    write_csv(
        out_dir / "acceleration_contribution_summary.csv",
        contribution_rows,
        ["subset", "quantity"] + ACCEL_STAT_FIELDS,
    )

    plot_selection(out_dir, by_type_rows)
    plot_weight_distributions(out_dir, weight_data, subset_masks, cli.plot_sample_limit)
    plot_acceleration_ratio(out_dir, accel_summaries)

    elapsed_seconds = time.time() - start_time
    abs_data_path = stable_absolute_path(cli.data)
    abs_checkpoint_path = stable_absolute_path(cli.checkpoint)
    abs_out_dir = stable_absolute_path(out_dir)
    local_rerun_command = (
        f"python analyze_routing_v2.py --checkpoint {cli.checkpoint} "
        f"--data {cli.data} --out-dir {cli.out_dir} "
        f"--batch-size {cli.batch_size} --device {cli.device}"
    )
    if cli.drop_last:
        local_rerun_command += " --drop-last"
    if cli.max_batches is not None:
        local_rerun_command += f" --max-batches {cli.max_batches}"

    rerun_command = local_rerun_command
    if "microsoft" in platform.platform().lower() and str(Path.cwd()).startswith("/mnt/"):
        rerun_command = (
            f'wsl.exe -d Ubuntu-24.04-D --cd "{Path.cwd()}" '
            f"-e /home/codex/venvs/unipe/bin/python "
            f"analyze_routing_v2.py --checkpoint {cli.checkpoint} "
            f"--data {cli.data} --out-dir {cli.out_dir} "
            f"--batch-size {cli.batch_size} --device {cli.device}"
        )
        if cli.drop_last:
            rerun_command += " --drop-last"
        if cli.max_batches is not None:
            rerun_command += f" --max-batches {cli.max_batches}"

    metadata = {
        "analysis": "routing_analysis_v2",
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": t.__version__,
        "cuda_available": bool(t.cuda.is_available()),
        "device": str(device),
        "data_path": cli.data,
        "checkpoint_path": cli.checkpoint,
        "output_dir": cli.out_dir,
        "data_path_absolute": abs_data_path,
        "checkpoint_path_absolute": abs_checkpoint_path,
        "output_dir_absolute": abs_out_dir,
        "sample_count": int(usable_samples),
        "total_data_samples": int(total_data_samples),
        "drop_last": bool(cli.drop_last),
        "max_batches": cli.max_batches,
        "batch_size": int(cli.batch_size),
        "model_file": "model/model_MoE_gru_new.py",
        "loader_file": "loader2.py",
        "type_mapping": {
            str(type_id): {
                "label": TYPE_LABELS[type_id],
                "family": TYPE_FAMILIES[type_id],
                "expert_model": "ACC" if type_id in ACC_TYPES else "IDM",
            }
            for type_id in ORDERED_TYPES
        },
        "routing_modes": {
            "soft_all": "Aggregate all four candidate accelerations with raw cf_probs.",
            "hard_top1": "Use only argmax(cf_probs).",
            "top2": "Use top-2 raw probabilities, renormalized over the selected pair.",
            "family_top2": "Choose ACC or IDM family by total family mass, then renormalize within that family.",
            "oracle_top1": "Diagnostic only; use ground-truth type.",
        },
        "average_definitions": {
            "avg_4h": "mean of 0.5s, 1.0s, 1.5s, and 2.0s RMSE",
            "avg_20step": "mean of per-step RMSE over all 20 prediction steps",
        },
        "rmse_aggregation": "global SSE over selected samples followed by square root",
        "acceleration_contribution_state_basis": (
            "candidate accelerations for contribution diagnostics are computed at "
            "the official Top-2 rollout state for each sample and step"
        ),
        "acceleration_quantile_method": (
            f"fixed-bin streaming histograms with {cli.accel_hist_bins} bins; "
            "mean/std/min/max are exact"
        ),
        "quality_checks": {
            "retraining_performed": False,
            "normal_modes_use_ground_truth": False,
            "oracle_top1_uses_ground_truth": True,
            "oracle_top1_diagnostic_only": True,
            "top2_aggregates_accelerations_not_parameters": True,
            "averages_labeled_separately": True,
            "sample_counts_reported": True,
        },
        "rerun_command": rerun_command,
        "rerun_command_inside_current_shell": local_rerun_command,
        "generated_files": [
            "metadata.json",
            "implementation_check.md",
            "official_top2_eval.csv",
            "routing_selection_summary.csv",
            "routing_selection_by_type.csv",
            "hard_top1_confusion_matrix.csv",
            "top2_pair_counts.csv",
            "routing_weight_summary.csv",
            "cross_family_weight_summary.csv",
            "low_confidence_weight_summary.csv",
            "routing_rmse_by_mode_and_subset.csv",
            "acceleration_contribution_summary.csv",
            "candidate_acceleration_stats.csv",
            "routing_analysis_v2.md",
            "selection_error_and_rescue_rates.png",
            "weight_distribution_top1_top2.png",
            "cross_family_second_weight_distribution.png",
            "cross_family_acceleration_contribution_ratio.png",
        ],
    }

    write_json(out_dir / "metadata.json", metadata)
    (out_dir / "implementation_check.md").write_text(
        make_implementation_check(metadata), encoding="utf-8"
    )
    report = make_report(
        metadata,
        official_rows,
        summary_rows,
        by_type_rows,
        rmse_rows,
        weight_rows,
        contribution_rows,
    )
    (out_dir / "routing_analysis_v2.md").write_text(report, encoding="utf-8")

    print(f"Routing analysis v2 complete: {out_dir}")
    print(f"Samples evaluated: {usable_samples}")
    print(f"Elapsed seconds: {elapsed_seconds:.1f}")


if __name__ == "__main__":
    main()
