import argparse
import csv
import os
import random

import numpy as np
import torch as t
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import loader2 as lo
import model.presimnet as model
from config import args, device, learning_rate


class HighSimMemmapDataset(Dataset):
    def __init__(self, npy_file, hist_length=20, fut_length=20):
        self.data = np.load(npy_file, mmap_mode="r")
        self.hist_length = hist_length
        self.fut_length = fut_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        hist_acc_diff = window[:, 5] - window[:, 7]
        hist = np.column_stack(
            (window[: self.hist_length, 4:], hist_acc_diff[: self.hist_length])
        ).astype(np.float32, copy=False)
        next_v = window[self.hist_length - 1 :, 4].astype(np.float32, copy=False)
        fut = window[self.hist_length :, [8, 6, 11, 12]].astype(np.float32, copy=False)
        cf_type = np.zeros(4, dtype=np.float32)
        cf_type[int(window[0, 3])] = 1.0

        return (
            t.from_numpy(hist),
            t.from_numpy(next_v),
            t.from_numpy(fut),
            t.from_numpy(cf_type),
        )


def param_distribution_loss(acc_params, idm_params, cf_probs):
    param_ranges = {
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

    total_loss = 0.0
    for i in range(2):
        params = acc_params[i].squeeze(1)
        acc_loss = 0.0
        for j, (min_val, max_val) in enumerate(param_ranges["acc"]):
            scale = 10.0 if j == 2 else 1.0
            param_value = params[:, j]
            acc_loss += t.mean(t.relu(min_val - param_value) + t.relu(param_value - max_val)) / scale
        total_loss += cf_probs[:, i].mean() * acc_loss

    for i in range(2):
        params = idm_params[i].squeeze(1)
        idm_loss = 0.0
        for j, (min_val, max_val) in enumerate(param_ranges["idm"]):
            scale = 10.0 if j == 4 else 1.0
            param_value = params[:, j]
            idm_loss += t.mean(t.relu(min_val - param_value) + t.relu(param_value - max_val)) / scale
        total_loss += cf_probs[:, i + 2].mean() * idm_loss

    return total_loss


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def find_last_linear(module):
    if hasattr(module, "layers"):
        children = module.layers
    else:
        children = module
    for layer in reversed(children):
        if isinstance(layer, nn.Linear):
            return layer
    raise RuntimeError("No Linear layer found in expert head.")


def initialize_expert_biases(encoder):
    with t.no_grad():
        for i, expert in enumerate(encoder.parameter_head.all_experts):
            last_layer = find_last_linear(expert)
            if i < 2:
                last_layer.bias.data[0] = 0.23
                last_layer.bias.data[1] = 0.07
                last_layer.bias.data[2] = 25.0
                last_layer.bias.data[3] = 2.0
                last_layer.bias.data[4] = 1.6
                last_layer.bias.data[5] = 0.2
            else:
                last_layer.bias.data[0] = 2.0
                last_layer.bias.data[1] = 1.6
                last_layer.bias.data[2] = 1.67
                last_layer.bias.data[3] = 1.5
                last_layer.bias.data[4] = 25.0
                last_layer.bias.data[5] = 4.0


def default_setting(run_args):
    return "ed{}_inl{}_ol{}_drop{}_tl{}_nh{}_od{}_gama{}_qv{}_nt{}_gru_new_2-top2".format(
        run_args["encoder_size"],
        run_args["in_length"],
        run_args["out_length"],
        run_args["dropout"],
        run_args["transformer_layer"],
        run_args["n_head"],
        run_args["out_dim"],
        run_args["gamma"],
        int(run_args["query_var_flag"]),
        run_args["num_task"],
    )


def build_dataset(cli, run_args):
    if cli.data_loading == "memory":
        return lo.HighSimDataset(cli.train_data, run_args["in_length"], run_args["out_length"])
    return HighSimMemmapDataset(cli.train_data, run_args["in_length"], run_args["out_length"])


def parse_args():
    parser = argparse.ArgumentParser(description="Train PreSimNet MoE with Top-2 expert aggregation.")
    parser.add_argument("--train-data", default="../data/train_data.npy")
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--last-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--data-loading", choices=("mmap", "memory"), default="mmap")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    cli = parse_args()

    random.seed(30)
    np.random.seed(72)
    t.manual_seed(72)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(72)

    run_args = dict(args)
    run_args["device"] = device
    run_args["train_flag"] = True
    run_args["gamma"] = cli.gamma
    run_args["epoch"] = cli.epochs
    run_args["last_epoch"] = cli.last_epoch
    run_args["batch_size"] = cli.batch_size
    run_args["num_worker"] = cli.num_workers
    run_args["aggregation_mode"] = "top2"
    run_args["route_top_k"] = 2

    setting = default_setting(run_args)
    run_args["path"] = cli.checkpoint_dir or os.path.join("checkpoints", setting)
    run_args["l_path"] = cli.result_dir or os.path.join("results", setting)
    os.makedirs(run_args["path"], exist_ok=True)
    os.makedirs(run_args["l_path"], exist_ok=True)

    encoder = model.Encoder(run_args)
    initialize_expert_biases(encoder)
    predictor = model.predictor(run_args)

    encoder = encoder.to(device)
    encoder.train()

    train_dataset = build_dataset(cli, run_args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=run_args["batch_size"],
        shuffle=True,
        num_workers=run_args["num_worker"],
        pin_memory=t.cuda.is_available(),
        drop_last=True,
    )

    optimizer = optim.Adam(encoder.parameters(), lr=cli.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=run_args["epoch"], last_epoch=-1)

    if run_args["last_epoch"] != 0:
        checkpoint = t.load(
            os.path.join(run_args["path"], f"epoch{run_args['last_epoch']}_e.tar"),
            map_location=device,
        )
        encoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    log_path = os.path.join(run_args["l_path"], "train_log.csv")
    write_header = not os.path.exists(log_path) or run_args["last_epoch"] == 0
    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "direct_pos",
                    "dynamic_gap",
                    "dynamic_vel",
                    "cf_type",
                    "param_dist",
                    "total",
                    "batches",
                ]
            )

        print(f"Device: {device}")
        print(f"Aggregation: {run_args['aggregation_mode']} k={run_args['route_top_k']}")
        print(f"Checkpoint dir: {run_args['path']}")
        print(f"Result dir: {run_args['l_path']}")
        print(f"Data loading: {cli.data_loading}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Trainable parameters: {count_parameters(encoder):,}")

        for epoch in range(run_args["last_epoch"], run_args["epoch"]):
            print(f"epoch: {epoch + 1} lr {optimizer.param_groups[0]['lr']}")
            epoch_losses = {
                "direct_pos": 0.0,
                "dynamic_gap": 0.0,
                "dynamic_vel": 0.0,
                "cf_type": 0.0,
                "param_dist": 0.0,
                "total": 0.0,
            }
            batch_count = 0

            for idx, data in enumerate(tqdm(train_loader)):
                hist, nextv, fut, cf_type = data
                hist = hist.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                nextv = nextv.to(device, non_blocking=True)
                cf_type = cf_type.to(device, non_blocking=True)
                veh_state = hist[:, -1, [4, 2]]

                model_outputs = encoder(hist)
                predictions = predictor.forward(model_outputs, nextv, veh_state)

                direct_pred = predictions["direct_pred"]
                dynamic_pred = predictions["dynamic_pred"]
                cf_type_pred = model_outputs["cf_type_pred"]
                params_dict = model_outputs["params"]
                cf_probs = params_dict["cf_probs"]
                route_probs = predictor.route_probabilities(cf_probs)

                direct_pos_loss = t.nn.MSELoss()(direct_pred.squeeze(-1), fut[:, :, 2])
                dynamic_gap_loss = t.nn.MSELoss()(dynamic_pred[:, :, 0], fut[:, :, 0])
                dynamic_vel_loss = t.nn.MSELoss()(dynamic_pred[:, :, 1], fut[:, :, 1])
                cf_type_loss = t.nn.CrossEntropyLoss()(cf_type_pred, cf_type)
                param_dist_loss = param_distribution_loss(
                    params_dict["acc_params"],
                    params_dict["idm_params"],
                    route_probs,
                )

                total_loss = (
                    direct_pos_loss
                    + dynamic_gap_loss
                    + dynamic_vel_loss
                    + cf_type_loss
                    + 0.01 * param_dist_loss
                )

                if t.isnan(total_loss).any():
                    raise RuntimeError("NaN detected in total loss.")

                optimizer.zero_grad()
                total_loss.backward()
                t.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_losses["direct_pos"] += direct_pos_loss.item()
                epoch_losses["dynamic_gap"] += dynamic_gap_loss.item()
                epoch_losses["dynamic_vel"] += dynamic_vel_loss.item()
                epoch_losses["cf_type"] += cf_type_loss.item()
                epoch_losses["param_dist"] += param_dist_loss.item()
                epoch_losses["total"] += total_loss.item()
                batch_count += 1

                if cli.max_batches is not None and batch_count >= cli.max_batches:
                    break

            scheduler.step()
            avg = {name: value / batch_count for name, value in epoch_losses.items()}
            print(f"Epoch {epoch + 1} Average Losses:")
            for name, value in avg.items():
                print(f"{name}: {value:.6f}")

            t.save(
                {
                    "epoch": epoch,
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg["total"],
                    "aggregation_mode": run_args["aggregation_mode"],
                    "route_top_k": run_args["route_top_k"],
                },
                os.path.join(run_args["path"], f"epoch{epoch + 1}_e.tar"),
            )
            writer.writerow(
                [
                    epoch + 1,
                    optimizer.param_groups[0]["lr"],
                    avg["direct_pos"],
                    avg["dynamic_gap"],
                    avg["dynamic_vel"],
                    avg["cf_type"],
                    avg["param_dist"],
                    avg["total"],
                    batch_count,
                ]
            )
            log_file.flush()


if __name__ == "__main__":
    main()
