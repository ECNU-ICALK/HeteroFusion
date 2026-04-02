#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import MixDatasetBuilder  # noqa: E402
from main import load_adapter_conf_safe, resolve_pipeline_config  # noqa: E402
from src.losses import compute_rdm_reg  # noqa: E402
from src.trainer import HeteroFusionTrainer, WeightPatcher  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run LR range test for a HeteroFusion config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the training yaml config.")
    parser.add_argument("--task_index", type=int, default=0, help="Task index in config['tasks'] to test.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Min learning rate.")
    parser.add_argument("--max_lr", type=float, default=2e-3, help="Max learning rate.")
    parser.add_argument("--optim_steps", type=int, default=40, help="Number of optimizer steps to scan.")
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps used in LR test. Defaults to config training setting.",
    )
    parser.add_argument("--beta", type=float, default=0.98, help="Smoothing factor for loss.")
    parser.add_argument("--diverge_th", type=float, default=4.0, help="Early stop threshold over best smoothed loss.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm. <=0 disables clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for LR test logs. Defaults to logs/lr_range_test/<timestamp>.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_adapter_state(adapter_dir: str):
    safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safe_path):
        return load_file(safe_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter model found under: {adapter_dir}")


def build_trainer(config_path: str, task_index: int):
    with open(config_path, "r", encoding="utf-8") as f:
        config = resolve_pipeline_config(yaml.safe_load(f), config_path)

    tasks = config["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task_index={task_index} out of range (num_tasks={len(tasks)}).")
    task = tasks[task_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading base model: {config['base_model_path']}")
    base_model_raw = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    data_builder = MixDatasetBuilder(config, config["base_model_path"])
    current_target_lora_path = config["initial_target_lora"]
    print(f"[INFO] Initial target LoRA: {current_target_lora_path}")

    peft_config = load_adapter_conf_safe(current_target_lora_path)
    model = PeftModel.from_pretrained(
        base_model_raw,
        current_target_lora_path,
        config=peft_config,
        is_trainable=False,
        adapter_name="default",
    )
    model.eval()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tgt_state = load_adapter_state(current_target_lora_path)
    dataloader, _ = data_builder.get_mixed_dataloader(task["datasets"])

    source_paths = task.get("source_lora_paths")
    if not source_paths:
        single = task.get("source_lora_path")
        source_paths = [single] if single else []
    if not source_paths:
        raise ValueError("No source LoRA path(s) found in task config.")

    trainer_config = {
        "output_dir": os.path.join(config["output_dir"], "lr_range_probe"),
        "model": {"transfer_ratio": task.get("transfer_ratio", 1.0)},
        "training": task["training"],
    }

    trainer = HeteroFusionTrainer(
        base_model=model,
        config=trainer_config,
        device=device,
        train_dataloader=dataloader,
        target_lora_state=tgt_state,
        source_lora_paths=source_paths,
        lpka_cache={},
    )
    return trainer


def compute_one_batch_losses(trainer: HeteroFusionTrainer, batch: dict):
    model_device = trainer.base_model.device
    batch = {k: v.to(model_device) for k, v in batch.items()}

    full_patch_dict = {}
    module_map_ref = {}
    total_rdm_loss = 0.0

    for group in trainer.active_groups.values():
        hypernet = group["hypernet"]
        inputs = group["inputs"]

        if group.get("batched", False):
            delta_flat, _, tgt_emb, src_emb = hypernet(
                inputs["tgt_flat"], inputs["tgt_s"], inputs["src_flat"], inputs["src_s"]
            )
            rdm_t = compute_rdm_reg(tgt_emb, trainer.mu_target, trainer.sigma_target, trainer.num_projections)
            rdm_s = compute_rdm_reg(src_emb, trainer.mu_target, trainer.sigma_target, trainer.num_projections)
            total_rdm_loss += rdm_t + rdm_s
        else:
            outs = []
            per_item_rdm = []
            for tf, ts, sf, ss in zip(inputs["tgt_flat"], inputs["tgt_s"], inputs["src_flat"], inputs["src_s"]):
                d, _, te, se = hypernet(tf.unsqueeze(0), ts.unsqueeze(0), sf.unsqueeze(0), ss.unsqueeze(0))
                outs.append(d.squeeze(0))
                rdm_t_i = compute_rdm_reg(te, trainer.mu_target, trainer.sigma_target, trainer.num_projections)
                rdm_s_i = compute_rdm_reg(se, trainer.mu_target, trainer.sigma_target, trainer.num_projections)
                per_item_rdm.append(rdm_t_i + rdm_s_i)
            delta_flat = outs
            if per_item_rdm:
                total_rdm_loss += torch.stack(per_item_rdm).mean()

        for i, mod_data in enumerate(group["modules"]):
            clean_name = mod_data["name"]
            ref_module = mod_data["ref_module"]
            info = group["reconstruct_info"][i]

            start_idx = info["num_blocks_A"]
            count = info["num_blocks_B"]
            delta_B = delta_flat[i][start_idx : start_idx + count]

            curr_delta = delta_B.view(-1, trainer.block_size, hypernet.rank)
            curr_delta = curr_delta.view(-1, hypernet.rank)
            curr_delta = curr_delta[: info["orig_rows"], :]

            if hasattr(ref_module.lora_B, "default"):
                orig_B = ref_module.lora_B.default.weight
            else:
                orig_B = ref_module.lora_B.weight

            if group["alpha"].dtype != curr_delta.dtype:
                group["alpha"].data = group["alpha"].data.to(curr_delta.dtype)

            update_term = group["alpha"] * curr_delta
            full_patch_dict[clean_name] = orig_B + update_term.to(orig_B.device)
            module_map_ref[clean_name] = ref_module

    if not full_patch_dict:
        raise RuntimeError("No patched LoRA modules found for LR test.")

    with WeightPatcher(module_map_ref, full_patch_dict):
        outputs = trainer.base_model(**batch)
        lm_loss = outputs.loss
        avg_rdm_loss = total_rdm_loss / max(1, len(trainer.active_groups))
        total_loss = lm_loss + trainer.lambda_reg * avg_rdm_loss

    return total_loss, lm_loss, avg_rdm_loss


def get_next_batch(dataloader, iterator):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch, iterator


def lr_at_step(min_lr: float, max_lr: float, step: int, total_steps: int):
    if total_steps <= 1:
        return max_lr
    ratio = step / float(total_steps - 1)
    return min_lr * ((max_lr / min_lr) ** ratio)


def suggest_lr(records):
    best_row = min(records, key=lambda x: x["smoothed_loss"])
    best_lr = best_row["lr"]
    min_loss = best_row["smoothed_loss"]

    near_valley = [r for r in records if r["smoothed_loss"] <= min_loss * 1.1]
    if near_valley:
        valley_low = min(r["lr"] for r in near_valley)
        valley_high = max(r["lr"] for r in near_valley)
    else:
        valley_low = best_lr
        valley_high = best_lr

    recommended = max(valley_low, min(best_lr * 0.5, valley_high))
    return {
        "best_step": best_row["step"],
        "best_lr_by_min_loss": best_lr,
        "valley_low_lr": valley_low,
        "valley_high_lr": valley_high,
        "recommended_lr": recommended,
    }


def write_outputs(out_dir: str, records: list, summary: dict):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "lr_range_records.csv")
    json_path = os.path.join(out_dir, "lr_range_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "lr", "loss", "lm_loss", "rdm_loss", "smoothed_loss"],
        )
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.min_lr <= 0 or args.max_lr <= 0 or args.max_lr <= args.min_lr:
        raise ValueError("Require 0 < min_lr < max_lr.")
    if args.optim_steps <= 0:
        raise ValueError("optim_steps must be > 0.")

    print("[INFO] Building trainer...")
    trainer = build_trainer(args.config_path, args.task_index)
    accum_steps = args.accum_steps or int(trainer.config["training"].get("gradient_accumulation_steps", 8))
    print(f"[INFO] LR scan setup: min_lr={args.min_lr:.3e}, max_lr={args.max_lr:.3e}, optim_steps={args.optim_steps}, accum_steps={accum_steps}")

    dataloader_iter = iter(trainer.dataloader)
    optimizer = trainer.optimizer
    optimizer.zero_grad()

    records = []
    smoothed = None
    best_smoothed = float("inf")
    completed_steps = 0
    t0 = time.time()

    for opt_step in range(args.optim_steps):
        lr = lr_at_step(args.min_lr, args.max_lr, opt_step, args.optim_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        total_loss_acc = 0.0
        lm_loss_acc = 0.0
        rdm_loss_acc = 0.0

        for _ in range(accum_steps):
            batch, dataloader_iter = get_next_batch(trainer.dataloader, dataloader_iter)
            total_loss, lm_loss, rdm_loss = compute_one_batch_losses(trainer, batch)

            (total_loss / accum_steps).backward()

            total_loss_acc += float(total_loss.detach().item())
            lm_loss_acc += float(lm_loss.detach().item())
            rdm_loss_acc += float(rdm_loss.detach().item()) if isinstance(rdm_loss, torch.Tensor) else float(rdm_loss)

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainer.param_groups, args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss_acc / accum_steps
        avg_lm = lm_loss_acc / accum_steps
        avg_rdm = rdm_loss_acc / accum_steps

        if smoothed is None:
            smoothed = avg_loss
        else:
            smoothed = args.beta * smoothed + (1.0 - args.beta) * avg_loss

        row = {
            "step": opt_step + 1,
            "lr": lr,
            "loss": avg_loss,
            "lm_loss": avg_lm,
            "rdm_loss": avg_rdm,
            "smoothed_loss": smoothed,
        }
        records.append(row)
        completed_steps += 1

        if math.isfinite(smoothed) and smoothed < best_smoothed:
            best_smoothed = smoothed

        print(
            f"[LR-TEST] step={opt_step + 1:03d}/{args.optim_steps} "
            f"lr={lr:.3e} loss={avg_loss:.4f} lm={avg_lm:.4f} rdm={avg_rdm:.4f} smooth={smoothed:.4f}"
        )

        diverged = (
            (not math.isfinite(smoothed))
            or ((opt_step + 1) > 10 and best_smoothed < float("inf") and smoothed > args.diverge_th * best_smoothed)
        )
        if diverged:
            print("[WARN] Early stop: loss diverged.")
            break

    elapsed = time.time() - t0
    if not records:
        raise RuntimeError("LR range test failed: no records generated.")

    recommendation = suggest_lr(records)
    summary = {
        "config_path": os.path.abspath(args.config_path),
        "task_index": args.task_index,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "optim_steps_requested": args.optim_steps,
        "optim_steps_completed": completed_steps,
        "accum_steps": accum_steps,
        "beta": args.beta,
        "diverge_th": args.diverge_th,
        "elapsed_sec": elapsed,
        "recommendation": recommendation,
    }

    out_dir = args.output_dir
    if not out_dir:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(PROJECT_ROOT, "logs", "lr_range_test", ts)

    csv_path, json_path = write_outputs(out_dir, records, summary)
    print("[RESULT] Recommendation:")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
    print(f"[RESULT] Records CSV: {csv_path}")
    print(f"[RESULT] Summary JSON: {json_path}")

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
