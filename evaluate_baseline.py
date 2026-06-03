import argparse
import csv
import importlib
import os
import time

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader2 as lo
from config import args as base_args
from config import device


BASELINE_MODULES = {
    "seq2seq": "model.model_seq2seq_baseline",
    "transformer": "model.model_transformer_baseline",
    "cslstm": "model.model_cslstm_baseline",
    "stdan": "model.model_stdan_baseline",
    "bat": "model.model_bat_baseline",
    "hltp": "model.model_hltp_baseline",
}


def rmse_batch_sum(pred, target, mask):
    err = (pred - target).pow(2) * mask
    return t.sqrt(t.sum(err, dim=0)), t.sqrt(t.sum(mask, dim=0))


def evaluate_checkpoint(
    model_name,
    checkpoint,
    data_path,
    output_path,
    batch_size=512,
    num_workers=0,
):
    run_args = dict(base_args)
    run_args["train_flag"] = False
    run_args["batch_size"] = batch_size
    run_args["num_worker"] = num_workers

    baseline_module = importlib.import_module(BASELINE_MODULES[model_name])
    baseline_model = baseline_module.Seq2SeqBaseline(run_args).to(device)
    predictor = baseline_module.predictor(run_args)

    checkpoint_data = t.load(checkpoint, map_location=device)
    baseline_model.load_state_dict(checkpoint_data["model_state_dict"])
    baseline_model.eval()

    dataset = lo.HighSimDataset(data_path, run_args["in_length"], run_args["out_length"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=t.cuda.is_available(),
        drop_last=True,
    )

    loss_vals = t.zeros(run_args["out_length"], 2, device=device)
    counts = t.zeros(run_args["out_length"], 2, device=device)
    pos_loss_vals = t.zeros(run_args["out_length"], 1, device=device)
    pos_counts = t.zeros(run_args["out_length"], 1, device=device)
    total_tp = t.zeros(run_args["cf_type"], device=device)
    total_fp = t.zeros(run_args["cf_type"], device=device)
    total_fn = t.zeros(run_args["cf_type"], device=device)
    elapsed = 0.0

    with t.no_grad():
        for hist, nextv, fut, cf_type in tqdm(loader, desc=f"Evaluate {model_name}"):
            hist = hist.to(device, non_blocking=True)
            nextv = nextv.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            cf_type = cf_type.to(device, non_blocking=True)
            veh_state = hist[:, -1, [4, 2]]

            start = time.time()
            outputs = baseline_model(hist)
            predictions = predictor.forward(outputs, nextv, veh_state)
            elapsed += time.time() - start

            dynamic_pred = predictions["dynamic_pred"]
            direct_pred = predictions["direct_pred"]
            cf_type_pred = outputs["cf_type_pred"]

            mask = t.ones_like(fut[:, :, :2])
            pos_mask = t.ones_like(fut[:, :, 2:3])

            loss, count = rmse_batch_sum(dynamic_pred, fut[:, :, :2], mask)
            pos_loss, pos_count = rmse_batch_sum(direct_pred, fut[:, :, 2:3], pos_mask)
            loss_vals += loss.detach()
            counts += count.detach()
            pos_loss_vals += pos_loss.detach()
            pos_counts += pos_count.detach()

            predicted = t.argmax(cf_type_pred, dim=1)
            target = t.argmax(cf_type, dim=1)
            for class_id in range(run_args["cf_type"]):
                total_tp[class_id] += ((predicted == class_id) & (target == class_id)).sum()
                total_fp[class_id] += ((predicted == class_id) & (target != class_id)).sum()
                total_fn[class_id] += ((predicted != class_id) & (target == class_id)).sum()

    dynamic_rmse = loss_vals / counts.clamp_min(1e-12)
    pos_rmse = pos_loss_vals / pos_counts.clamp_min(1e-12)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *[f"{(i + 1) * 0.1:.1f}s" for i in range(run_args["out_length"])], "avg"])
        writer.writerow(["gap_rmse", *dynamic_rmse[:, 0].detach().cpu().tolist(), dynamic_rmse[:, 0].mean().item()])
        writer.writerow(["vel_rmse", *dynamic_rmse[:, 1].detach().cpu().tolist(), dynamic_rmse[:, 1].mean().item()])
        writer.writerow(["pos_rmse", *pos_rmse[:, 0].detach().cpu().tolist(), pos_rmse[:, 0].mean().item()])
        writer.writerow([])
        writer.writerow(["class", "precision", "recall", "f1"])
        for class_id in range(run_args["cf_type"]):
            writer.writerow(
                [
                    class_id,
                    precision[class_id].item(),
                    recall[class_id].item(),
                    f1[class_id].item(),
                ]
            )

    print(f"Saved: {output_path}")
    print(f"Average inference time: {elapsed / max(len(loader), 1):.4f}s/batch")
    return {
        "dynamic_rmse": dynamic_rmse.detach().cpu(),
        "position_rmse": pos_rmse.detach().cpu(),
        "precision": precision.detach().cpu(),
        "recall": recall.detach().cpu(),
        "f1": f1.detach().cpu(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PreSimNet baseline checkpoint.")
    parser.add_argument("--model", choices=sorted(BASELINE_MODULES), default="hltp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="../data/test_data.npy")
    parser.add_argument("--out", default="results/baseline_test_loss.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    evaluate_checkpoint(
        model_name=cli.model,
        checkpoint=cli.checkpoint,
        data_path=cli.data,
        output_path=cli.out,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
    )
