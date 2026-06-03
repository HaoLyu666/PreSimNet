from __future__ import annotations

import argparse
import os

import numpy as np
import torch as t
import torch.nn.functional as F

from config import args as base_args
import model.presimnet as presimnet


TYPE_LABELS = np.array(["AV-HV", "AV-AV", "HV-HV", "HV-AV"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PreSimNet inference on an .npy file.")
    parser.add_argument("--data", default="data/sample/test_data_sample.npy")
    parser.add_argument("--checkpoint", default="checkpoints/presimnet_top2/epoch21_e.tar")
    parser.add_argument("--out", default="outputs/sample_predictions.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if t.cuda.is_available() else "cpu")
    return parser.parse_args()


def iter_batches(data_path: str, batch_size: int, in_length: int, max_samples: int | None):
    data = np.load(data_path, mmap_mode="r")
    total = len(data) if max_samples is None else min(len(data), max_samples)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        window = data[start:end]
        hist_acc_diff = window[:, :, 5] - window[:, :, 7]
        hist = np.concatenate(
            [window[:, :in_length, 4:], hist_acc_diff[:, :in_length, None]],
            axis=2,
        ).astype(np.float32, copy=False)
        next_v = window[:, in_length - 1 :, 4].astype(np.float32, copy=False)
        yield hist, next_v


def main() -> None:
    cli = parse_args()
    device = t.device(cli.device)

    run_args = dict(base_args)
    run_args["device"] = device
    run_args["train_flag"] = False
    run_args["aggregation_mode"] = "top2"
    run_args["route_top_k"] = 2

    model = presimnet.PreSimNet(run_args).to(device)
    checkpoint = t.load(cli.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    simulator = presimnet.predictor(run_args)

    position_pred = []
    dynamic_pred = []
    cf_type_prob = []
    cf_type_id = []

    with t.no_grad():
        for hist_np, next_v_np in iter_batches(
            cli.data, cli.batch_size, run_args["in_length"], cli.max_samples
        ):
            hist = t.from_numpy(hist_np).to(device)
            next_v = t.from_numpy(next_v_np).to(device)
            veh_state = hist[:, -1, [4, 2]]

            outputs = model(hist)
            predictions = simulator.forward(outputs, next_v, veh_state)
            probs = F.softmax(outputs["cf_type_pred"], dim=-1)

            position_pred.append(predictions["direct_pred"].detach().cpu().numpy())
            dynamic_pred.append(predictions["dynamic_pred"].detach().cpu().numpy())
            cf_type_prob.append(probs.detach().cpu().numpy())
            cf_type_id.append(t.argmax(probs, dim=-1).detach().cpu().numpy())

    position_pred_np = np.concatenate(position_pred, axis=0)
    dynamic_pred_np = np.concatenate(dynamic_pred, axis=0)
    cf_type_prob_np = np.concatenate(cf_type_prob, axis=0)
    cf_type_id_np = np.concatenate(cf_type_id, axis=0)

    out_dir = os.path.dirname(cli.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        cli.out,
        position_pred=position_pred_np,
        dynamic_pred=dynamic_pred_np,
        cf_type_prob=cf_type_prob_np,
        cf_type_id=cf_type_id_np,
        cf_type_label=TYPE_LABELS[cf_type_id_np],
    )

    print(f"samples: {position_pred_np.shape[0]}")
    print(f"position_pred: {position_pred_np.shape}")
    print(f"dynamic_pred: {dynamic_pred_np.shape}")
    print(f"saved: {cli.out}")


if __name__ == "__main__":
    main()
