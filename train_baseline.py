import argparse
import importlib
import os

import torch as t
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader2 as lo
from config import args as base_args
from config import device, learning_rate
from evaluate_baseline import evaluate_checkpoint


BASELINE_MODULES = {
    "seq2seq": "model.model_seq2seq_baseline",
    "transformer": "model.model_transformer_baseline",
    "cslstm": "model.model_cslstm_baseline",
    "stdan": "model.model_stdan_baseline",
    "bat": "model.model_bat_baseline",
    "hltp": "model.model_hltp_baseline",
}


def load_baseline_module(name):
    return importlib.import_module(BASELINE_MODULES[name])


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PreSimNet baseline model.")
    parser.add_argument("--model", choices=sorted(BASELINE_MODULES), default="hltp")
    parser.add_argument("--train-data", default="../data/train_data.npy")
    parser.add_argument("--test-data", default="../data/test_data.npy")
    parser.add_argument("--epochs", type=int, default=base_args["epoch"])
    parser.add_argument("--last-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=base_args["batch_size"])
    parser.add_argument("--num-workers", type=int, default=base_args["num_worker"])
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    cli = parse_args()
    run_args = dict(base_args)
    run_args["train_flag"] = True
    run_args["batch_size"] = cli.batch_size
    run_args["num_worker"] = cli.num_workers
    run_args["epoch"] = cli.epochs
    run_args["last_epoch"] = cli.last_epoch

    checkpoint_dir = cli.checkpoint_dir or os.path.join("checkpoints", f"{cli.model}_baseline")
    result_dir = cli.result_dir or os.path.join("results", f"{cli.model}_baseline")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    baseline_module = load_baseline_module(cli.model)
    baseline_model = baseline_module.Seq2SeqBaseline(run_args).to(device)
    baseline_model.train()

    train_dataset = lo.HighSimDataset(
        cli.train_data, run_args["in_length"], run_args["out_length"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=cli.num_workers,
        pin_memory=t.cuda.is_available(),
        drop_last=True,
    )

    optimizer = optim.Adam(baseline_model.parameters(), lr=cli.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cli.epochs, last_epoch=-1)

    if cli.last_epoch != 0:
        checkpoint = t.load(
            os.path.join(checkpoint_dir, f"baseline_epoch{cli.last_epoch}.tar"),
            map_location=device,
        )
        baseline_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Baseline: {cli.model}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Trainable parameters: {count_parameters(baseline_model):,}")

    for epoch in range(cli.last_epoch, cli.epochs):
        print(f"epoch: {epoch + 1} lr {optimizer.param_groups[0]['lr']}")
        losses = {"position": 0.0, "cf_type": 0.0}
        batch_count = 0

        for data in tqdm(train_loader):
            hist, _nextv, fut, cf_type = data
            hist = hist.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            cf_type = cf_type.to(device, non_blocking=True)

            outputs = baseline_model(hist)
            position_loss = t.nn.MSELoss()(outputs["position_pred"].squeeze(-1), fut[:, :, 2])
            cf_type_loss = t.nn.CrossEntropyLoss()(outputs["cf_type_pred"], cf_type)
            total_loss = position_loss + cf_type_loss

            optimizer.zero_grad()
            total_loss.backward()
            t.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=2.0)
            optimizer.step()

            losses["position"] += position_loss.item()
            losses["cf_type"] += cf_type_loss.item()
            batch_count += 1
            if cli.max_batches is not None and batch_count >= cli.max_batches:
                break

        scheduler.step()
        avg = {name: value / batch_count for name, value in losses.items()}
        print(f"Epoch {epoch + 1} Average Losses:")
        for name, value in avg.items():
            print(f"{name}: {value:.6f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"baseline_epoch{epoch + 1}.tar")
        t.save(
            {
                "epoch": epoch,
                "model_state_dict": baseline_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg,
                "baseline": cli.model,
            },
            checkpoint_path,
        )

        if cli.eval_every and (epoch + 1) % cli.eval_every == 0:
            evaluate_checkpoint(
                model_name=cli.model,
                checkpoint=checkpoint_path,
                data_path=cli.test_data,
                output_path=os.path.join(result_dir, "baseline_test_loss.csv"),
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
            )


if __name__ == "__main__":
    main()
