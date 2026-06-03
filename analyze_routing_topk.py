from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from tqdm import tqdm

from config import args as base_args
import model.model_MoE_gru_new as model


TYPE_LABELS = {
    0: "AV-HV",
    1: "AV-AV",
    2: "HV-HV",
    3: "HV-AV",
}
ORDERED_TYPES = [1, 0, 3, 2]
ACC_TYPES = {0, 1}
IDM_TYPES = {2, 3}


class EvalStyleAccumulator:
    """Replicates the batch-wise RMSE aggregation used in evaluate_new.py."""

    def __init__(self, out_length: int, dims: int, device: t.device):
        self.loss_vals = t.zeros(out_length, dims, device=device)
        self.counts = t.zeros(out_length, dims, device=device)
        self.samples = 0

    def update(self, pred: t.Tensor, gt: t.Tensor, sample_mask: t.Tensor) -> None:
        if not bool(sample_mask.any()):
            return

        pred_sel = pred[sample_mask]
        gt_sel = gt[sample_mask]
        mask = t.ones_like(gt_sel)

        err = (pred_sel - gt_sel).pow(2) * mask
        loss_val = t.sqrt(t.sum(err, dim=0))
        counts = t.sqrt(t.sum(mask, dim=0))

        self.loss_vals += loss_val.detach()
        self.counts += counts.detach()
        self.samples += int(sample_mask.sum().item())

    def result(self) -> np.ndarray:
        out = self.loss_vals / (self.counts + 1e-12)
        return out.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hard and Top-2 expert routing errors and RMSE."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "presimnet_top2/"
            "epoch21_e.tar"
        ),
        help="Checkpoint path for the full PreSimNet model.",
    )
    parser.add_argument(
        "--data",
        default="../data/test_data.npy",
        help="Test data .npy path.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/routing_analysis",
        help="Directory for figures and CSV files.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda:0" if t.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the final incomplete batch, matching the legacy evaluation scripts.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit.",
    )
    parser.add_argument(
        "--plot-sample-limit",
        type=int,
        default=200000,
        help="Maximum number of samples used in distribution plots.",
    )
    return parser.parse_args()


def make_batch_iterator(
    data_path: str,
    batch_size: int,
    in_length: int,
    out_length: int,
    drop_last: bool,
    max_batches: int | None,
):
    data = np.load(data_path, mmap_mode="r")
    total = len(data)
    if drop_last:
        usable = (total // batch_size) * batch_size
    else:
        usable = total

    batch_count = (usable + batch_size - 1) // batch_size
    if max_batches is not None:
        batch_count = min(batch_count, max_batches)
        usable = min(usable, batch_count * batch_size)

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


def top2_probs(probs: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    top_vals, top_idx = t.topk(probs, k=2, dim=1)
    norm_vals = top_vals / (top_vals.sum(dim=1, keepdim=True) + 1e-12)
    routed = t.zeros_like(probs)
    routed.scatter_(1, top_idx, norm_vals)
    return routed, top_idx, top_vals, norm_vals


def family_constrained_probs(probs: t.Tensor) -> t.Tensor:
    routed = t.zeros_like(probs)
    acc_mass = probs[:, 0:2].sum(dim=1, keepdim=True)
    idm_mass = probs[:, 2:4].sum(dim=1, keepdim=True)
    use_acc = (acc_mass >= idm_mass).squeeze(1)

    acc_weights = probs[:, 0:2] / (acc_mass + 1e-12)
    idm_weights = probs[:, 2:4] / (idm_mass + 1e-12)
    routed[use_acc, 0:2] = acc_weights[use_acc]
    routed[~use_acc, 2:4] = idm_weights[~use_acc]
    return routed


def routed_outputs(model_outputs: dict, route_probs: t.Tensor) -> dict:
    params = dict(model_outputs["params"])
    params["cf_probs"] = route_probs
    out = dict(model_outputs)
    out["params"] = params
    return out


def add_count_labels(ax, bars, counts, rates=None, y_pad=0.5):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if rates is None:
            text = f"{int(counts[i])}"
        else:
            text = f"{int(counts[i])}\n{rates[i]:.2f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_pad,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_selection_summary(out_dir: str, stats: dict) -> None:
    totals = stats["total_by_type"]
    hard_wrong = stats["hard_wrong_by_type"]
    top2_miss = stats["top2_miss_by_type"]
    rescue = stats["top2_rescue_by_type"]
    cross_family = stats["cross_family_by_type"]

    labels = [TYPE_LABELS[i] for i in ORDERED_TYPES]
    total_arr = np.array([totals[i] for i in ORDERED_TYPES], dtype=float)
    hard_arr = np.array([hard_wrong[i] for i in ORDERED_TYPES], dtype=float)
    miss_arr = np.array([top2_miss[i] for i in ORDERED_TYPES], dtype=float)
    rescue_arr = np.array([rescue[i] for i in ORDERED_TYPES], dtype=float)
    cross_arr = np.array([cross_family[i] for i in ORDERED_TYPES], dtype=float)

    hard_rate = np.divide(hard_arr, total_arr, out=np.zeros_like(hard_arr), where=total_arr > 0) * 100
    miss_rate = np.divide(miss_arr, total_arr, out=np.zeros_like(miss_arr), where=total_arr > 0) * 100
    cross_rate = np.divide(cross_arr, total_arr, out=np.zeros_like(cross_arr), where=total_arr > 0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    x = np.arange(len(labels))
    width = 0.36

    bars1 = axes[0].bar(x - width / 2, hard_rate, width, label="Hard Top-1")
    bars2 = axes[0].bar(x + width / 2, miss_rate, width, label="Top-2 miss")
    y0_max = max(float(np.max(hard_rate)), float(np.max(miss_rate)), 0.1)
    axes[0].set_ylim(0, y0_max * 1.28)
    add_count_labels(axes[0], bars1, hard_arr, hard_rate, y_pad=y0_max * 0.035)
    add_count_labels(axes[0], bars2, miss_arr, miss_rate, y_pad=y0_max * 0.035)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Expert-selection error rate (%)")
    axes[0].set_title("(a) Hard vs Top-2 selection errors")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    still_miss = miss_arr
    bars3 = axes[1].bar(x, rescue_arr, label="Recovered by Top-2")
    bars4 = axes[1].bar(x, still_miss, bottom=rescue_arr, label="Still missed by Top-2")
    y1_max = max(float(np.max(rescue_arr + still_miss)), 1.0)
    axes[1].set_ylim(0, y1_max * 1.18)
    for i, bar in enumerate(bars4):
        total_height = rescue_arr[i] + still_miss[i]
        if rescue_arr[i] > 0:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                rescue_arr[i] / 2,
                f"{int(rescue_arr[i])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            total_height + y1_max * 0.02,
            f"miss {int(still_miss[i])}\ntotal {int(total_height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Hard Top-1 error samples")
    axes[1].set_title("(b) Top-2 coverage of Top-1 errors")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    bars5 = axes[2].bar(x, cross_rate, color="#8f63b8")
    y2_max = max(float(np.max(cross_rate)), 0.1)
    axes[2].set_ylim(0, y2_max * 1.18)
    add_count_labels(axes[2], bars5, cross_arr, cross_rate, y_pad=y2_max * 0.025)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Cross-family Top-2 rate (%)")
    axes[2].set_title("(c) ACC-IDM mixed Top-2 pairs")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "routing_selection_analysis.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "routing_selection_analysis.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution(out_dir: str, weight_data: dict, plot_sample_limit: int) -> None:
    rng = np.random.default_rng(72)
    probs = weight_data["probs"]
    top_vals = weight_data["top_vals"]
    norm_vals = weight_data["norm_vals"]
    true_prob = weight_data["true_prob"]
    targets = weight_data["targets"]
    top1_correct = weight_data["top1_correct"].astype(bool)

    box_labels = [
        "Top-1 raw",
        "Top-2 raw",
        "Top-1 renorm",
        "Top-2 renorm",
        "True expert",
    ]

    def sample_indices(mask: np.ndarray) -> np.ndarray:
        pool = np.flatnonzero(mask)
        if pool.size > plot_sample_limit:
            return rng.choice(pool, size=plot_sample_limit, replace=False)
        return pool

    def make_box_data(idx: np.ndarray) -> list[np.ndarray]:
        return [
            top_vals[idx, 0],
            top_vals[idx, 1],
            norm_vals[idx, 0],
            norm_vals[idx, 1],
            true_prob[idx],
        ]

    mean_prob = np.zeros((4, 4), dtype=np.float64)
    for label in range(4):
        mask = targets == label
        if np.any(mask):
            mean_prob[label] = probs[mask].mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=300)

    panels = [
        (axes[0], sample_indices(np.ones_like(top1_correct, dtype=bool)), "(a) All samples"),
        (axes[1], sample_indices(~top1_correct), "(b) Hard Top-1 errors"),
    ]

    for ax, idx, title in panels:
        box_data = make_box_data(idx)
        violin = ax.violinplot(box_data, showmeans=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#4f8fc9")
            body.set_alpha(0.45)
        violin["cmeans"].set_color("#303030")
        ax.boxplot(
            box_data,
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "#303030"},
            medianprops={"color": "#d62728"},
        )
        ax.set_xticks(np.arange(1, len(box_labels) + 1))
        ax.set_xticklabels(box_labels, rotation=22, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("Softmax probability / selected routing weight")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    ordered_rows = ORDERED_TYPES
    ordered_cols = ORDERED_TYPES
    heat = mean_prob[np.ix_(ordered_rows, ordered_cols)]
    im = axes[2].imshow(heat, vmin=0, vmax=1, cmap="YlGnBu")
    axes[2].set_xticks(np.arange(4))
    axes[2].set_xticklabels([TYPE_LABELS[i] for i in ordered_cols])
    axes[2].set_yticks(np.arange(4))
    axes[2].set_yticklabels([TYPE_LABELS[i] for i in ordered_rows])
    axes[2].set_xlabel("Expert probability")
    axes[2].set_ylabel("Ground-truth type")
    axes[2].set_title("(c) Mean softmax probabilities")
    for r in range(4):
        for c in range(4):
            axes[2].text(c, r, f"{heat[r, c]:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "selected_weight_distribution.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "selected_weight_distribution.pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_weights(weight_data: dict) -> list[dict]:
    groups = {
        "all": np.ones_like(weight_data["targets"], dtype=bool),
        "top1_correct": weight_data["top1_correct"],
        "top1_wrong": ~weight_data["top1_correct"],
        "top2_hit": weight_data["top2_hit"],
        "top2_miss": ~weight_data["top2_hit"],
    }
    metrics = {
        "top1_raw": weight_data["top_vals"][:, 0],
        "top2_raw": weight_data["top_vals"][:, 1],
        "top1_renorm": weight_data["norm_vals"][:, 0],
        "top2_renorm": weight_data["norm_vals"][:, 1],
        "true_expert_prob": weight_data["true_prob"],
    }

    rows = []
    for group_name, mask in groups.items():
        for metric_name, values in metrics.items():
            selected = values[mask]
            if selected.size == 0:
                continue
            rows.append(
                {
                    "group": group_name,
                    "metric": metric_name,
                    "n": int(selected.size),
                    "mean": float(np.mean(selected)),
                    "std": float(np.std(selected)),
                    "p05": float(np.quantile(selected, 0.05)),
                    "p25": float(np.quantile(selected, 0.25)),
                    "median": float(np.quantile(selected, 0.50)),
                    "p75": float(np.quantile(selected, 0.75)),
                    "p95": float(np.quantile(selected, 0.95)),
                }
            )
    return rows


def summarize_cross_family_weights(weight_data: dict) -> list[dict]:
    probs = weight_data["probs"]
    targets = weight_data["targets"]
    top_idx = np.argsort(-probs, axis=1)[:, :2]
    top_vals = np.take_along_axis(probs, top_idx, axis=1)
    norm_vals = top_vals / (top_vals.sum(axis=1, keepdims=True) + 1e-12)

    first_is_acc = np.isin(top_idx[:, 0], list(ACC_TYPES))
    second_is_acc = np.isin(top_idx[:, 1], list(ACC_TYPES))
    cross_family = first_is_acc != second_is_acc

    rows = []
    for label in ORDERED_TYPES:
        type_mask = targets == label
        mask = type_mask & cross_family
        if not np.any(type_mask):
            continue
        rows.append(
            {
                "type_id": label,
                "type_label": TYPE_LABELS[label],
                "total_samples": int(type_mask.sum()),
                "cross_family_count": int(mask.sum()),
                "cross_family_rate": float(mask.sum() / type_mask.sum()),
                "top1_raw_mean_when_cross": float(np.mean(top_vals[mask, 0]))
                if np.any(mask)
                else 0.0,
                "top2_raw_mean_when_cross": float(np.mean(top_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_raw_median_when_cross": float(np.median(top_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_renorm_mean_when_cross": float(np.mean(norm_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_renorm_median_when_cross": float(np.median(norm_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
            }
        )

    if len(targets) > 0:
        mask = cross_family
        rows.append(
            {
                "type_id": "all",
                "type_label": "ALL",
                "total_samples": int(len(targets)),
                "cross_family_count": int(mask.sum()),
                "cross_family_rate": float(mask.mean()),
                "top1_raw_mean_when_cross": float(np.mean(top_vals[mask, 0]))
                if np.any(mask)
                else 0.0,
                "top2_raw_mean_when_cross": float(np.mean(top_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_raw_median_when_cross": float(np.median(top_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_renorm_mean_when_cross": float(np.mean(norm_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
                "top2_renorm_median_when_cross": float(np.median(norm_vals[mask, 1]))
                if np.any(mask)
                else 0.0,
            }
        )

    return rows


def rmse_rows(accumulators: dict, out_length: int) -> list[dict]:
    checkpoints = {
        "0.5s": 4,
        "1.0s": 9,
        "1.5s": 14,
        "2.0s": 19,
    }
    rows = []
    for mode_name, subset_accs in accumulators.items():
        for subset_name, acc in subset_accs.items():
            values = acc.result()
            gap = values[:, 0]
            vel = values[:, 1]
            row = {
                "mode": mode_name,
                "subset": subset_name,
                "n_samples": acc.samples,
            }
            for label, idx in checkpoints.items():
                row[f"gap_rmse_{label}"] = float(gap[idx])
                row[f"vel_rmse_{label}"] = float(vel[idx])
            row["gap_rmse_avg"] = float(np.mean(gap[:out_length]))
            row["vel_rmse_avg"] = float(np.mean(vel[:out_length]))
            rows.append(row)
    return rows


def main() -> None:
    cli = parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    device = t.device(cli.device)
    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False
    run_args["batch_size"] = cli.batch_size

    if device.type == "cuda":
        t.set_float32_matmul_precision("high")

    encoder = model.Encoder(run_args).to(device)
    checkpoint = t.load(cli.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    predictor = model.predictor(run_args)

    out_length = run_args["out_length"]
    cf_type = run_args["cf_type"]

    total_by_type = np.zeros(cf_type, dtype=np.int64)
    hard_wrong_by_type = np.zeros(cf_type, dtype=np.int64)
    top2_miss_by_type = np.zeros(cf_type, dtype=np.int64)
    top2_rescue_by_type = np.zeros(cf_type, dtype=np.int64)
    cross_family_by_type = np.zeros(cf_type, dtype=np.int64)
    confusion = np.zeros((cf_type, cf_type), dtype=np.int64)
    top2_pair_counts = np.zeros((cf_type, cf_type), dtype=np.int64)

    prob_sum_by_type = np.zeros((cf_type, cf_type), dtype=np.float64)

    weight_chunks = defaultdict(list)
    modes = ["soft_all", "hard_top1", "top2", "family_top2"]
    subsets = ["all", "top1_wrong", "top2_rescue", "top2_miss"]
    accumulators = {
        mode_name: {
            subset_name: EvalStyleAccumulator(out_length, 2, device)
            for subset_name in subsets
        }
        for mode_name in modes
    }

    data = np.load(cli.data, mmap_mode="r")
    total = len(data)
    del data
    usable = (total // cli.batch_size) * cli.batch_size if cli.drop_last else total
    if cli.max_batches is not None:
        usable = min(usable, cli.max_batches * cli.batch_size)
    total_batches = (usable + cli.batch_size - 1) // cli.batch_size

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
            iterator, total=total_batches, desc="Analyzing routing"
        ):
            hist = t.as_tensor(hist_np, device=device)
            nextv = t.as_tensor(nextv_np, device=device)
            fut = t.as_tensor(fut_np, device=device)
            target = t.as_tensor(target_np, dtype=t.long, device=device)

            model_outputs = encoder(hist)
            logits = model_outputs["cf_type_pred"]
            probs = F.softmax(logits, dim=1)
            top2_route, top_idx, top_vals, norm_vals = top2_probs(probs)
            top1_idx = top_idx[:, 0]
            hard_route = one_hot_from_indices(top1_idx, cf_type)
            family_route = family_constrained_probs(probs)

            target_cpu = target_np
            top1_cpu = top1_idx.detach().cpu().numpy()
            top2_cpu = top_idx.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            top_vals_cpu = top_vals.detach().cpu().numpy()
            norm_vals_cpu = norm_vals.detach().cpu().numpy()

            top1_correct_cpu = top1_cpu == target_cpu
            top2_hit_cpu = np.any(top2_cpu == target_cpu[:, None], axis=1)
            cross_family_cpu = np.array(
                [
                    (top2_cpu[i, 0] in ACC_TYPES and top2_cpu[i, 1] in IDM_TYPES)
                    or (top2_cpu[i, 0] in IDM_TYPES and top2_cpu[i, 1] in ACC_TYPES)
                    for i in range(len(target_cpu))
                ],
                dtype=bool,
            )

            for label in range(cf_type):
                label_mask = target_cpu == label
                total_by_type[label] += int(label_mask.sum())
                hard_wrong_by_type[label] += int((label_mask & ~top1_correct_cpu).sum())
                top2_miss_by_type[label] += int((label_mask & ~top2_hit_cpu).sum())
                top2_rescue_by_type[label] += int(
                    (label_mask & ~top1_correct_cpu & top2_hit_cpu).sum()
                )
                cross_family_by_type[label] += int((label_mask & cross_family_cpu).sum())
                if np.any(label_mask):
                    prob_sum_by_type[label] += probs_cpu[label_mask].sum(axis=0)

            np.add.at(confusion, (target_cpu, top1_cpu), 1)
            np.add.at(top2_pair_counts, (top2_cpu[:, 0], top2_cpu[:, 1]), 1)

            true_prob_cpu = probs_cpu[np.arange(len(target_cpu)), target_cpu]
            weight_chunks["probs"].append(probs_cpu.astype(np.float32))
            weight_chunks["top_vals"].append(top_vals_cpu.astype(np.float32))
            weight_chunks["norm_vals"].append(norm_vals_cpu.astype(np.float32))
            weight_chunks["true_prob"].append(true_prob_cpu.astype(np.float32))
            weight_chunks["targets"].append(target_cpu.astype(np.int16))
            weight_chunks["top1_correct"].append(top1_correct_cpu)
            weight_chunks["top2_hit"].append(top2_hit_cpu)

            route_dict = {
                "soft_all": probs,
                "hard_top1": hard_route,
                "top2": top2_route,
                "family_top2": family_route,
            }
            subset_masks = {
                "all": t.ones_like(target, dtype=t.bool),
                "top1_wrong": top1_idx != target,
                "top2_rescue": (top1_idx != target)
                & ((top_idx[:, 0] == target) | (top_idx[:, 1] == target)),
                "top2_miss": (top_idx[:, 0] != target) & (top_idx[:, 1] != target),
            }

            veh_state = hist[:, -1, [4, 2]]
            gt_dynamic = fut[:, :, :2]
            for mode_name, route_probs in route_dict.items():
                predictions = predictor.forward(
                    routed_outputs(model_outputs, route_probs), nextv, veh_state
                )
                dynamic_pred = predictions["dynamic_pred"]
                for subset_name, subset_mask in subset_masks.items():
                    accumulators[mode_name][subset_name].update(
                        dynamic_pred, gt_dynamic, subset_mask
                    )

    stats = {
        "total_by_type": total_by_type,
        "hard_wrong_by_type": hard_wrong_by_type,
        "top2_miss_by_type": top2_miss_by_type,
        "top2_rescue_by_type": top2_rescue_by_type,
        "cross_family_by_type": cross_family_by_type,
    }

    summary_rows = []
    for label in ORDERED_TYPES:
        total_label = int(total_by_type[label])
        hard_wrong = int(hard_wrong_by_type[label])
        top2_miss = int(top2_miss_by_type[label])
        rescue = int(top2_rescue_by_type[label])
        cross_family = int(cross_family_by_type[label])
        summary_rows.append(
            {
                "type_id": label,
                "type_label": TYPE_LABELS[label],
                "total_samples": total_label,
                "hard_wrong_count": hard_wrong,
                "hard_wrong_rate": hard_wrong / max(total_label, 1),
                "top2_miss_count": top2_miss,
                "top2_miss_rate": top2_miss / max(total_label, 1),
                "top2_rescue_count": rescue,
                "top2_rescue_rate_among_hard_errors": rescue / max(hard_wrong, 1),
                "cross_family_top2_count": cross_family,
                "cross_family_top2_rate": cross_family / max(total_label, 1),
            }
        )

    write_csv(
        os.path.join(cli.out_dir, "routing_selection_summary.csv"),
        summary_rows,
        [
            "type_id",
            "type_label",
            "total_samples",
            "hard_wrong_count",
            "hard_wrong_rate",
            "top2_miss_count",
            "top2_miss_rate",
            "top2_rescue_count",
            "top2_rescue_rate_among_hard_errors",
            "cross_family_top2_count",
            "cross_family_top2_rate",
        ],
    )

    confusion_rows = []
    for true_label in ORDERED_TYPES:
        row = {"true_type": TYPE_LABELS[true_label]}
        for pred_label in ORDERED_TYPES:
            row[f"pred_{TYPE_LABELS[pred_label]}"] = int(confusion[true_label, pred_label])
        confusion_rows.append(row)
    write_csv(
        os.path.join(cli.out_dir, "hard_top1_confusion_matrix.csv"),
        confusion_rows,
        ["true_type"] + [f"pred_{TYPE_LABELS[i]}" for i in ORDERED_TYPES],
    )

    pair_rows = []
    for first in ORDERED_TYPES:
        for second in ORDERED_TYPES:
            pair_rows.append(
                {
                    "top1": TYPE_LABELS[first],
                    "top2": TYPE_LABELS[second],
                    "count": int(top2_pair_counts[first, second]),
                    "cross_family": int(
                        (first in ACC_TYPES and second in IDM_TYPES)
                        or (first in IDM_TYPES and second in ACC_TYPES)
                    ),
                }
            )
    write_csv(
        os.path.join(cli.out_dir, "top2_pair_counts.csv"),
        pair_rows,
        ["top1", "top2", "count", "cross_family"],
    )

    rmse_table = rmse_rows(accumulators, out_length)
    write_csv(
        os.path.join(cli.out_dir, "routing_rmse_eval_style.csv"),
        rmse_table,
        [
            "mode",
            "subset",
            "n_samples",
            "gap_rmse_0.5s",
            "vel_rmse_0.5s",
            "gap_rmse_1.0s",
            "vel_rmse_1.0s",
            "gap_rmse_1.5s",
            "vel_rmse_1.5s",
            "gap_rmse_2.0s",
            "vel_rmse_2.0s",
            "gap_rmse_avg",
            "vel_rmse_avg",
        ],
    )

    weight_data = {
        key: np.concatenate(chunks, axis=0) for key, chunks in weight_chunks.items()
    }
    weight_data["mean_prob_by_true"] = np.divide(
        prob_sum_by_type,
        total_by_type[:, None],
        out=np.zeros_like(prob_sum_by_type),
        where=total_by_type[:, None] > 0,
    )

    np.savez_compressed(
        os.path.join(cli.out_dir, "routing_weight_data.npz"),
        **weight_data,
        total_by_type=total_by_type,
        confusion=confusion,
        top2_pair_counts=top2_pair_counts,
    )

    weight_rows = summarize_weights(weight_data)
    write_csv(
        os.path.join(cli.out_dir, "weight_distribution_summary.csv"),
        weight_rows,
        [
            "group",
            "metric",
            "n",
            "mean",
            "std",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
        ],
    )

    cross_weight_rows = summarize_cross_family_weights(weight_data)
    write_csv(
        os.path.join(cli.out_dir, "cross_family_weight_summary.csv"),
        cross_weight_rows,
        [
            "type_id",
            "type_label",
            "total_samples",
            "cross_family_count",
            "cross_family_rate",
            "top1_raw_mean_when_cross",
            "top2_raw_mean_when_cross",
            "top2_raw_median_when_cross",
            "top2_renorm_mean_when_cross",
            "top2_renorm_median_when_cross",
        ],
    )

    plot_selection_summary(cli.out_dir, stats)
    plot_weight_distribution(cli.out_dir, weight_data, cli.plot_sample_limit)

    print("\nRouting selection summary:")
    for row in summary_rows:
        print(
            f"{row['type_label']}: "
            f"hard_wrong={row['hard_wrong_count']} ({row['hard_wrong_rate'] * 100:.3f}%), "
            f"top2_miss={row['top2_miss_count']} ({row['top2_miss_rate'] * 100:.3f}%), "
            f"rescue={row['top2_rescue_count']}, "
            f"cross_family={row['cross_family_top2_count']} "
            f"({row['cross_family_top2_rate'] * 100:.3f}%)"
        )
    print(f"\nOutputs saved to: {cli.out_dir}")


if __name__ == "__main__":
    main()
