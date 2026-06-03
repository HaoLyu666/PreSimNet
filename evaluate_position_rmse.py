from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch as t
from tqdm import tqdm

from config import args as base_args
import model.model_MoE_gru_new as model


CHECKPOINTS = {
    "soft_all_original": (
        "checkponint/"
        "ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2/"
        "epoch21_e.tar"
    ),
    "top2_trained": (
        "checkponint/"
        "ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2-top2/"
        "epoch21_e.tar"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct position prediction RMSE.")
    parser.add_argument("--data", default="../data/test_data.npy")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--out", default="fig/vis/position_rmse_compare.csv")
    parser.add_argument("--device", default="cuda:0" if t.cuda.is_available() else "cpu")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def make_batch_iterator(data_path, batch_size, in_length, drop_last, max_batches):
    data = np.load(data_path, mmap_mode="r")
    total = len(data)
    usable = (total // batch_size) * batch_size if drop_last else total
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
        fut_pos = window[:, in_length:, 11:12].astype(np.float32, copy=False)
        yield hist, fut_pos


def eval_checkpoint(label, checkpoint_path, run_args, device, cli):
    encoder = model.Encoder(run_args).to(device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()

    loss_vals = t.zeros(run_args["out_length"], 1, device=device)
    counts = t.zeros(run_args["out_length"], 1, device=device)
    samples = 0

    with t.no_grad():
        for hist_np, fut_pos_np in tqdm(
            make_batch_iterator(
                cli.data,
                cli.batch_size,
                run_args["in_length"],
                cli.drop_last,
                cli.max_batches,
            ),
            desc=f"Position RMSE {label}",
        ):
            hist = t.from_numpy(hist_np).to(device)
            fut_pos = t.from_numpy(fut_pos_np).to(device)
            pred = encoder(hist)["position_pred"]
            mask = t.ones_like(fut_pos)
            err = (pred - fut_pos).pow(2) * mask
            loss_vals += t.sqrt(t.sum(err, dim=0))
            counts += t.sqrt(t.sum(mask, dim=0))
            samples += hist.shape[0]

    rmse = (loss_vals / (counts + 1e-12)).detach().cpu().numpy().reshape(-1)
    horizons = [4, 9, 14, 19]
    row = {"model": label, "checkpoint": checkpoint_path, "n_samples": samples}
    for idx in horizons:
        row[f"pos_rmse_{(idx + 1) * 0.1:.1f}s"] = float(rmse[idx])
    row["pos_rmse_avg"] = float(rmse.mean())
    return row


def main():
    cli = parse_args()
    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    device = t.device(cli.device)
    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False

    rows = [
        eval_checkpoint(label, path, run_args, device, cli)
        for label, path in CHECKPOINTS.items()
    ]
    fieldnames = [
        "model",
        "checkpoint",
        "n_samples",
        "pos_rmse_0.5s",
        "pos_rmse_1.0s",
        "pos_rmse_1.5s",
        "pos_rmse_2.0s",
        "pos_rmse_avg",
    ]
    with open(cli.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Saved to: {cli.out}")


if __name__ == "__main__":
    main()
