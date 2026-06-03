from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

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

HORIZONS = {
    "0.5s": 4,
    "1.0s": 9,
    "1.5s": 14,
    "2.0s": 19,
}

PRIMARY_GROUPS = [
    ("low_speed", "v_f(t) < 10 m/s"),
    ("medium_speed", "10 <= v_f(t) < 20 m/s"),
    ("high_speed", "v_f(t) >= 20 m/s"),
]

OPTIONAL_GROUPS = [
    ("high_20_30", "20 <= v_f(t) < 30 m/s"),
    ("very_high_30_plus", "v_f(t) >= 30 m/s"),
]

ALL_GROUP_NAMES = [name for name, _ in PRIMARY_GROUPS + OPTIONAL_GROUPS]


class LegacyRMSEAccumulator:
    """Batch-wise RMSE accumulator matching evaluate_new.py dxMSETest."""

    def __init__(self, out_length: int, device: t.device):
        self.loss_vals = t.zeros(out_length, 1, device=device)
        self.counts = t.zeros(out_length, 1, device=device)
        self.samples = 0

    def update(self, pred: t.Tensor, gt: t.Tensor, sample_mask: t.Tensor) -> None:
        n = int(sample_mask.sum().item())
        if n == 0:
            return
        pred_sel = pred[sample_mask]
        gt_sel = gt[sample_mask]
        mask = t.ones_like(gt_sel)
        acc = (pred_sel - gt_sel).pow(2) * mask
        self.loss_vals += t.pow(t.sum(acc, dim=0), 0.5).detach()
        self.counts += t.pow(t.sum(mask, dim=0), 0.5).detach()
        self.samples += n

    def result(self) -> np.ndarray:
        values = (self.loss_vals / (self.counts + 1e-12)).detach().cpu().numpy()
        values[self.counts.detach().cpu().numpy() == 0] = np.nan
        return values.reshape(-1)


class MetricBundle:
    def __init__(self, out_length: int, device: t.device, cf_type: int):
        self.pos = LegacyRMSEAccumulator(out_length, device)
        self.vel = LegacyRMSEAccumulator(out_length, device)
        self.gap = LegacyRMSEAccumulator(out_length, device)
        self.type_correct = np.zeros(cf_type, dtype=np.int64)
        self.type_total = np.zeros(cf_type, dtype=np.int64)

    @property
    def samples(self) -> int:
        return self.pos.samples

    def update(
        self,
        direct_pred: t.Tensor,
        dynamic_pred: t.Tensor,
        fut: t.Tensor,
        target: t.Tensor,
        predicted_type: t.Tensor,
        sample_mask: t.Tensor,
    ) -> None:
        self.pos.update(direct_pred, fut[:, :, 2:3], sample_mask)
        self.gap.update(dynamic_pred[:, :, 0:1], fut[:, :, 0:1], sample_mask)
        self.vel.update(dynamic_pred[:, :, 1:2], fut[:, :, 1:2], sample_mask)

        if not bool(sample_mask.any()):
            return
        target_np = target[sample_mask].detach().cpu().numpy()
        pred_np = predicted_type[sample_mask].detach().cpu().numpy()
        for type_id in range(len(self.type_total)):
            type_mask = target_np == type_id
            total = int(np.sum(type_mask))
            self.type_total[type_id] += total
            if total:
                self.type_correct[type_id] += int(np.sum(pred_np[type_mask] == type_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PreSimNet speed-stratified test-set evaluation materials."
    )
    parser.add_argument("--data", default="../data/test_data.npy")
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkponint/"
            "ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2/"
            "epoch21_e.tar"
        ),
    )
    parser.add_argument("--out-dir", default="fig/vis/speed_stratified_analysis")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the final partial batch. Default is drop_last=True, matching evaluate_new.py.",
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--routing-mode", default="top2", choices=("top2",))
    parser.add_argument("--route-top-k", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if t.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--optional-min-count", type=int, default=1000)
    return parser.parse_args()


def usable_sample_count(total: int, batch_size: int, keep_last: bool, max_batches: int | None) -> int:
    usable = total if keep_last else (total // batch_size) * batch_size
    if max_batches is not None:
        usable = min(usable, max_batches * batch_size)
    return usable


def batch_iterator(
    data_path: str,
    batch_size: int,
    in_length: int,
    out_length: int,
    keep_last: bool,
    max_batches: int | None,
):
    data = np.load(data_path, mmap_mode="r")
    usable = usable_sample_count(len(data), batch_size, keep_last, max_batches)
    for start in range(0, usable, batch_size):
        end = min(start + batch_size, usable)
        if end <= start:
            break
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


def group_masks(speed: t.Tensor) -> dict[str, t.Tensor]:
    return {
        "low_speed": speed < 10.0,
        "medium_speed": (speed >= 10.0) & (speed < 20.0),
        "high_speed": speed >= 20.0,
        "high_20_30": (speed >= 20.0) & (speed < 30.0),
        "very_high_30_plus": speed >= 30.0,
    }


def metric_values(bundle: MetricBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return bundle.pos.result(), bundle.vel.result(), bundle.gap.result()


def nanmean_or_blank(values: list[float] | np.ndarray) -> float | str:
    values_arr = np.asarray(values, dtype=np.float64)
    finite = values_arr[np.isfinite(values_arr)]
    if finite.size == 0:
        return ""
    return float(np.mean(finite))


def four_horizon_mean(values: np.ndarray) -> float | str:
    return nanmean_or_blank([values[idx] for idx in HORIZONS.values()])


def all_step_mean(values: np.ndarray) -> float | str:
    return nanmean_or_blank(values)


def type_accuracy_fields(bundle: MetricBundle) -> tuple[dict[str, float | str], float | str]:
    fields: dict[str, float | str] = {}
    accuracies = []
    for type_id, label in TYPE_LABELS.items():
        total = int(bundle.type_total[type_id])
        if total:
            acc = float(bundle.type_correct[type_id] / total)
            accuracies.append(acc)
            fields[f"type_acc_{label.replace('-', '_')}"] = acc
        else:
            fields[f"type_acc_{label.replace('-', '_')}"] = ""
    avg = float(np.mean(accuracies)) if accuracies else ""
    return fields, avg


def add_rmse_fields(row: dict, prefix: str, values: np.ndarray) -> None:
    for label, idx in HORIZONS.items():
        row[f"{prefix}_rmse_{label}"] = "" if math.isnan(values[idx]) else float(values[idx])
    row[f"{prefix}_rmse_avg_4h"] = four_horizon_mean(values)
    row[f"{prefix}_rmse_avg_20step"] = all_step_mean(values)


def summary_row(
    speed_group: str,
    speed_range: str,
    bundle: MetricBundle,
    sample_count_total: int,
    checkpoint_path: str,
    routing_mode: str,
    raw_test_sample_count: int,
    batch_size: int,
    drop_last: bool,
    dry_run: bool,
) -> dict:
    pos_values, vel_values, gap_values = metric_values(bundle)
    type_fields, type_acc_avg = type_accuracy_fields(bundle)
    row: dict = {
        "speed_group": speed_group,
        "speed_range": speed_range,
        "sample_count": bundle.samples,
        "sample_ratio": bundle.samples / sample_count_total if sample_count_total else 0.0,
    }
    add_rmse_fields(row, "pos", pos_values)
    add_rmse_fields(row, "vel", vel_values)
    add_rmse_fields(row, "gap", gap_values)
    row.update(type_fields)
    row["type_acc_avg"] = type_acc_avg
    row["checkpoint_path"] = checkpoint_path
    row["routing_mode"] = routing_mode
    row["sample_count_total"] = sample_count_total
    row["raw_test_sample_count"] = raw_test_sample_count
    row["batch_size"] = batch_size
    row["drop_last"] = drop_last
    row["dry_run"] = dry_run
    return row


def by_type_row(
    speed_group: str,
    speed_range: str,
    type_id: int,
    bundle: MetricBundle,
    sample_count_total: int,
    speed_group_count: int,
    checkpoint_path: str,
    routing_mode: str,
    batch_size: int,
    drop_last: bool,
    dry_run: bool,
) -> dict:
    pos_values, vel_values, gap_values = metric_values(bundle)
    type_total = int(bundle.type_total[type_id])
    type_correct = int(bundle.type_correct[type_id])
    row: dict = {
        "speed_group": speed_group,
        "speed_range": speed_range,
        "type_id": type_id,
        "type_label": TYPE_LABELS[type_id],
        "sample_count": bundle.samples,
        "sample_ratio_total": bundle.samples / sample_count_total if sample_count_total else 0.0,
        "sample_ratio_within_speed": bundle.samples / speed_group_count if speed_group_count else 0.0,
        "type_correct": type_correct,
        "type_acc": type_correct / type_total if type_total else "",
    }
    add_rmse_fields(row, "pos", pos_values)
    add_rmse_fields(row, "vel", vel_values)
    add_rmse_fields(row, "gap", gap_values)
    row["checkpoint_path"] = checkpoint_path
    row["routing_mode"] = routing_mode
    row["sample_count_total"] = sample_count_total
    row["batch_size"] = batch_size
    row["drop_last"] = drop_last
    row["dry_run"] = dry_run
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object, digits: int = 6) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    selected = rows[:max_rows] if max_rows is not None else rows
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if key.endswith("ratio") or "ratio" in key:
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    out_dir: Path,
    summary_rows: list[dict],
    by_type_rows: list[dict],
    included_optional: list[str],
    omitted_optional: list[str],
    raw_test_sample_count: int,
    sample_count_total: int,
    checkpoint_path: str,
    routing_mode: str,
    batch_size: int,
    drop_last: bool,
    dry_run: bool,
    max_batches: int | None,
) -> str:
    high_by_type = [
        row for row in by_type_rows if row["speed_group"] == "high_speed"
    ]
    high_by_type_sorted = sorted(
        high_by_type, key=lambda row: int(row["sample_count"]), reverse=True
    )

    lines = [
        "# Speed-Stratified PreSimNet Evaluation",
        "",
        "This report summarizes test-set speed-stratified evaluation materials only. It is not a reviewer response and is not manuscript text.",
        "",
        "## Evaluation setup",
        "",
        f"- Test data: `../data/test_data.npy`",
        f"- Raw test samples: `{raw_test_sample_count}`",
        f"- Evaluated samples: `{sample_count_total}`",
        f"- Batch size: `{batch_size}`",
        f"- Drop final incomplete batch: `{drop_last}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "- Model file: `model/model_MoE_gru_new.py`",
        f"- Routing mode: `{routing_mode}`",
        f"- Dry run: `{dry_run}`",
        f"- Max batches: `{max_batches if max_batches is not None else 'None'}`",
        "- Speed grouping feature: `hist[:, -1, 2]`, confirmed from `loader2.py` as the following-vehicle velocity at the last historical step.",
        "- No future speed is used for grouping.",
        "- No test data is used for checkpoint selection in this script.",
        "- RMSE uses the legacy `evaluate_new.py` batch-wise `dxMSETest` aggregation.",
        "- `avg_4h` is the mean of RMSE at 0.5s, 1.0s, 1.5s, and 2.0s. Use `avg_4h` for the paper table.",
        "- `avg_20step` is reported separately as a diagnostic and is not mixed with `avg_4h`.",
        "",
        "## Primary Speed Groups",
        "",
        markdown_table(
            [row for row in summary_rows if row["speed_group"] in {name for name, _ in PRIMARY_GROUPS}],
            [
                ("Group", "speed_group"),
                ("Range", "speed_range"),
                ("n", "sample_count"),
                ("ratio", "sample_ratio"),
                ("pos avg_4h", "pos_rmse_avg_4h"),
                ("vel avg_4h", "vel_rmse_avg_4h"),
                ("gap avg_4h", "gap_rmse_avg_4h"),
                ("type acc", "type_acc_avg"),
            ],
        ),
    ]

    if included_optional:
        lines.extend(
            [
                "",
                "## High-Speed Subgroups",
                "",
                "These optional rows subdivide `high_speed` and therefore overlap with that primary group.",
                "",
                markdown_table(
                    [row for row in summary_rows if row["speed_group"] in included_optional],
                    [
                        ("Group", "speed_group"),
                        ("Range", "speed_range"),
                        ("n", "sample_count"),
                        ("ratio", "sample_ratio"),
                        ("pos avg_4h", "pos_rmse_avg_4h"),
                        ("vel avg_4h", "vel_rmse_avg_4h"),
                        ("gap avg_4h", "gap_rmse_avg_4h"),
                        ("type acc", "type_acc_avg"),
                    ],
                ),
            ]
        )
    if omitted_optional:
        lines.extend(
            [
                "",
                "Optional subgroup(s) omitted because the evaluated sample count was below the configured threshold: "
                + ", ".join(omitted_optional)
                + ".",
            ]
        )

    lines.extend(
        [
            "",
            "## avg_20step Diagnostics",
            "",
            markdown_table(
                summary_rows,
                [
                    ("Group", "speed_group"),
                    ("pos avg_20step", "pos_rmse_avg_20step"),
                    ("vel avg_20step", "vel_rmse_avg_20step"),
                    ("gap avg_20step", "gap_rmse_avg_20step"),
                ],
            ),
            "",
            "## High-Speed Type Composition",
            "",
        ]
    )
    if high_by_type_sorted:
        lines.append(
            markdown_table(
                high_by_type_sorted,
                [
                    ("Type", "type_label"),
                    ("n", "sample_count"),
                    ("within high", "sample_ratio_within_speed"),
                    ("pos avg_4h", "pos_rmse_avg_4h"),
                    ("vel avg_4h", "vel_rmse_avg_4h"),
                    ("gap avg_4h", "gap_rmse_avg_4h"),
                    ("type acc", "type_acc"),
                ],
            )
        )
        dominant = high_by_type_sorted[0]
        lines.append("")
        lines.append(
            f"High-speed samples are most frequent in `{dominant['type_label']}` "
            f"({int(dominant['sample_count'])} samples, "
            f"{float(dominant['sample_ratio_within_speed']):.4f} of high-speed samples)."
        )
    else:
        lines.append("No high-speed samples were evaluated.")

    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- `{out_dir / 'speed_stratified_summary.csv'}`",
            f"- `{out_dir / 'speed_stratified_by_type.csv'}`",
            f"- `{out_dir / 'speed_stratified_report.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cli = parse_args()
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = np.load(cli.data, mmap_mode="r")
    raw_test_sample_count = int(len(data))
    sample_count_total = usable_sample_count(
        raw_test_sample_count,
        cli.batch_size,
        cli.keep_last,
        cli.max_batches,
    )
    del data

    device = t.device(cli.device)
    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False
    run_args["gamma"] = cli.gamma
    run_args["batch_size"] = cli.batch_size
    run_args["aggregation_mode"] = cli.routing_mode
    run_args["route_top_k"] = cli.route_top_k

    encoder = model.Encoder(run_args).to(device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    simulator = model.predictor(run_args)

    out_length = int(run_args["out_length"])
    cf_type = int(run_args["cf_type"])
    group_bundles = {
        group: MetricBundle(out_length, device, cf_type) for group in ALL_GROUP_NAMES
    }
    type_bundles = {
        (group, type_id): MetricBundle(out_length, device, cf_type)
        for group in ALL_GROUP_NAMES
        for type_id in TYPE_LABELS
    }

    total_batches = (
        sample_count_total + cli.batch_size - 1
    ) // cli.batch_size if sample_count_total else 0

    with t.inference_mode():
        iterator = batch_iterator(
            cli.data,
            cli.batch_size,
            int(run_args["in_length"]),
            out_length,
            cli.keep_last,
            cli.max_batches,
        )
        for hist_np, nextv_np, fut_np, target_np in tqdm(
            iterator, total=total_batches, desc="speed-stratified eval"
        ):
            hist = t.as_tensor(hist_np, device=device)
            nextv = t.as_tensor(nextv_np, device=device)
            fut = t.as_tensor(fut_np, device=device)
            target = t.as_tensor(target_np, dtype=t.long, device=device)

            veh_state = hist[:, -1, [4, 2]]
            speed = hist[:, -1, 2]
            outputs = encoder(hist)
            predictions = simulator.forward(outputs, nextv, veh_state)
            direct_pred = predictions["direct_pred"]
            dynamic_pred = predictions["dynamic_pred"]
            predicted_type = t.argmax(outputs["cf_type_pred"], dim=1)

            masks = group_masks(speed)
            for group, mask in masks.items():
                group_bundles[group].update(
                    direct_pred,
                    dynamic_pred,
                    fut,
                    target,
                    predicted_type,
                    mask,
                )
                for type_id in TYPE_LABELS:
                    type_mask = mask & (target == type_id)
                    type_bundles[(group, type_id)].update(
                        direct_pred,
                        dynamic_pred,
                        fut,
                        target,
                        predicted_type,
                        type_mask,
                    )

    included_groups: list[tuple[str, str]] = PRIMARY_GROUPS.copy()
    included_optional: list[str] = []
    omitted_optional: list[str] = []
    optional_ranges = dict(OPTIONAL_GROUPS)
    for group, speed_range in OPTIONAL_GROUPS:
        if group_bundles[group].samples >= cli.optional_min_count:
            included_groups.append((group, speed_range))
            included_optional.append(group)
        else:
            omitted_optional.append(group)

    summary_rows = [
        summary_row(
            group,
            speed_range,
            group_bundles[group],
            sample_count_total,
            str(checkpoint_path),
            cli.routing_mode,
            raw_test_sample_count,
            cli.batch_size,
            not cli.keep_last,
            cli.dry_run,
        )
        for group, speed_range in included_groups
    ]

    by_type_rows = []
    for group, speed_range in included_groups:
        speed_group_count = group_bundles[group].samples
        for type_id in TYPE_LABELS:
            by_type_rows.append(
                by_type_row(
                    group,
                    speed_range,
                    type_id,
                    type_bundles[(group, type_id)],
                    sample_count_total,
                    speed_group_count,
                    str(checkpoint_path),
                    cli.routing_mode,
                    cli.batch_size,
                    not cli.keep_last,
                    cli.dry_run,
                )
            )

    required_summary_fields = [
        "speed_group",
        "speed_range",
        "sample_count",
        "sample_ratio",
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
        "type_acc_avg",
        "checkpoint_path",
        "routing_mode",
        "sample_count_total",
    ]
    extra_summary_fields = [
        "pos_rmse_avg_20step",
        "vel_rmse_avg_20step",
        "gap_rmse_avg_20step",
        "type_acc_AV_HV",
        "type_acc_AV_AV",
        "type_acc_HV_HV",
        "type_acc_HV_AV",
        "raw_test_sample_count",
        "batch_size",
        "drop_last",
        "dry_run",
    ]
    summary_fields = required_summary_fields + [
        field for field in extra_summary_fields if field not in required_summary_fields
    ]

    by_type_fields = [
        "speed_group",
        "speed_range",
        "type_id",
        "type_label",
        "sample_count",
        "sample_ratio_total",
        "sample_ratio_within_speed",
        "type_correct",
        "type_acc",
        "pos_rmse_0.5s",
        "pos_rmse_1.0s",
        "pos_rmse_1.5s",
        "pos_rmse_2.0s",
        "pos_rmse_avg_4h",
        "pos_rmse_avg_20step",
        "vel_rmse_0.5s",
        "vel_rmse_1.0s",
        "vel_rmse_1.5s",
        "vel_rmse_2.0s",
        "vel_rmse_avg_4h",
        "vel_rmse_avg_20step",
        "gap_rmse_0.5s",
        "gap_rmse_1.0s",
        "gap_rmse_1.5s",
        "gap_rmse_2.0s",
        "gap_rmse_avg_4h",
        "gap_rmse_avg_20step",
        "checkpoint_path",
        "routing_mode",
        "sample_count_total",
        "batch_size",
        "drop_last",
        "dry_run",
    ]

    write_csv(out_dir / "speed_stratified_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "speed_stratified_by_type.csv", by_type_rows, by_type_fields)

    report = build_report(
        out_dir,
        summary_rows,
        by_type_rows,
        included_optional,
        omitted_optional,
        raw_test_sample_count,
        sample_count_total,
        str(checkpoint_path),
        cli.routing_mode,
        cli.batch_size,
        not cli.keep_last,
        cli.dry_run,
        cli.max_batches,
    )
    (out_dir / "speed_stratified_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {out_dir / 'speed_stratified_summary.csv'}")
    print(f"Wrote {out_dir / 'speed_stratified_by_type.csv'}")
    print(f"Wrote {out_dir / 'speed_stratified_report.md'}")


if __name__ == "__main__":
    main()
