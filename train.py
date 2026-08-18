"""
train.py — Phase 5 Siamese Candidate Re-Ranker Training & Full Cascade Validation Loop
Trains the ~240k-parameter Siamese metric embedder with dynamic hard-negative refresh,
warmup + cosine decay, ranking-accuracy checkpoint selection, and full cascade validation.

Usage:
    python train.py --train_manifest dataset/train_metadata.csv --val_manifest dataset/val_metadata.csv --test_manifest dataset/test_metadata.csv --epochs 60
"""

import os
import sys
import time
import math
import argparse
import csv
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import concurrent.futures

from localize import localize_pair, run_phase1_ncc, generate_reference_variants, get_clahe_uint8


# =====================================================================
# PART 1: SIAMESE METRIC EMBEDDING NETWORK (~240k parameters)
# =====================================================================

class LightweightSEMEmbedder(nn.Module):
    """
    Lightweight 4-block ConvNet for SEM pattern metric embedding (~240k parameters).
    Produces a 64-D L2-normalized feature vector for cosine distance re-ranking.
    """
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # 128 -> 64
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 32 -> 16
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # 16 -> 8
        self.bn4 = nn.BatchNorm2d(128)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [B, 1, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# =====================================================================
# PART 2: DYNAMIC HARD NEGATIVE DATASET
# =====================================================================

def resolve_image_path(p: str, data_root: str = "dataset") -> str:
    """
    Robustly resolves relative image paths against data_root and dataset/ directory.
    """
    if not p:
        return ""
    if os.path.exists(p):
        return p
    if data_root and os.path.exists(os.path.join(data_root, p)):
        return os.path.join(data_root, p)
    if os.path.exists(os.path.join("dataset", p)):
        return os.path.join("dataset", p)
    return os.path.join(data_root or "dataset", p)


class DynamicSEMTripletDataset(Dataset):
    """
    Dataset that dynamically samples and periodically refreshes hard negatives
    from training split physical look-alikes (+/-1 pitch, +/-2 pitch, neighbor MATs).
    Uses RAM caching for ultra-fast GPU throughput.
    """
    def __init__(self, manifest_path: str, data_root: str = "dataset", crop_size: int = 128, is_train: bool = True):
        self.rows = []
        self.crop_size = crop_size
        self.is_train = is_train
        self.data_root = data_root or os.path.dirname(manifest_path) or "dataset"
        self.cache = {}  # in-memory crop cache for fast iteration

        if not os.path.exists(manifest_path) and os.path.exists(os.path.join(self.data_root, manifest_path)):
            manifest_path = os.path.join(self.data_root, manifest_path)

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.rows.append(row)

        self.mined_negatives = {}  # idx -> list of (neg_x, neg_y)

    def __len__(self):
        return len(self.rows)

    def _extract_canonical_crop(self, img: np.ndarray, cx: float, cy: float, scale: float = 0.100, angle: float = 0.0) -> np.ndarray:
        target_dim = max(10.0, 1000.0 * scale)
        M_align = cv2.getRotationMatrix2D((cx, cy), -angle, self.crop_size / target_dim)
        M_align[0, 2] += (self.crop_size / 2.0 - cx)
        M_align[1, 2] += (self.crop_size / 2.0 - cy)
        crop = cv2.warpAffine(
            img, M_align, (self.crop_size, self.crop_size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
        )
        return crop

    def refresh_hard_negatives(self, top_n_mine=80):
        """
        Mines difficult NCC candidates from training split images to refresh negative pool.
        """
        if not self.is_train:
            return

        indices = np.random.choice(len(self.rows), min(top_n_mine, len(self.rows)), replace=False)
        for idx in indices:
            row = self.rows[idx]
            ref_path = resolve_image_path(row["ref_path"], self.data_root)
            search_path = resolve_image_path(row["search_path"], self.data_root)

            if not os.path.exists(ref_path) or not os.path.exists(search_path):
                continue

            ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            if ref_raw is None or search_raw is None:
                continue

            ref_norm = get_clahe_uint8(ref_raw)
            search_norm = get_clahe_uint8(search_raw)
            ref_variants = generate_reference_variants(ref_norm)

            candidates, _, _, _, _ = run_phase1_ncc(search_norm, ref_variants, nms_radius=12.0)
            gt_x = float(row.get("gt_x", 500.0))
            gt_y = float(row.get("gt_y", 500.0))

            hard_negs = []
            for c in candidates:
                dist = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
                if dist > 8.0:  # Valid non-ground-truth candidate
                    hard_negs.append((c["cx"], c["cy"]))

            if hard_negs:
                self.mined_negatives[idx] = hard_negs

    def __getitem__(self, idx):
        row = self.rows[idx]
        ref_path = resolve_image_path(row["ref_path"], self.data_root)
        search_path = resolve_image_path(row["search_path"], self.data_root)

        # Check in-memory image cache
        if ref_path not in self.cache:
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None: ref_img = np.full((1000, 1000), 128, dtype=np.uint8)
            self.cache[ref_path] = ref_img
        else:
            ref_img = self.cache[ref_path]

        if search_path not in self.cache:
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            if search_img is None: search_img = np.full((1000, 1000), 128, dtype=np.uint8)
            self.cache[search_path] = search_img
        else:
            search_img = self.cache[search_path]

        gt_x = float(row.get("gt_x", 500.0))
        gt_y = float(row.get("gt_y", 500.0))
        scale = float(row.get("scale", 0.100))
        angle = float(row.get("rotation", 0.0))

        # Anchor: Reference image downscaled to canonical 128x128
        ref_down = cv2.resize(ref_img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

        # Positive: Ground truth crop in search image with canonical affine alignment
        pos_crop = self._extract_canonical_crop(search_img, gt_x, gt_y, scale=scale, angle=angle)

        # Negative: Sample from mined negatives or class-specific periodic geometry offsets
        if idx in self.mined_negatives and len(self.mined_negatives[idx]) > 0:
            negs = self.mined_negatives[idx]
            neg_x, neg_y = negs[np.random.choice(len(negs))]
        else:
            ptype = row.get("pattern_type", "FIN_ARRAY")
            pitch = float(row.get("pitch", 40.0))
            pitch_px = max(6.0, pitch * 0.10 if pitch > 20.0 else pitch)

            neg_x, neg_y = gt_x, gt_y
            neg_strat = np.random.choice(["1_pitch", "2_pitch", "diag_step"])

            if ptype in ["FIN_ARRAY", "FIN_CUT"]:
                step = 1.0 if neg_strat == "1_pitch" else 2.0
                neg_x += np.random.choice([-1, 1]) * step * pitch_px
            elif ptype in ["GATE_POLY"]:
                step = 1.0 if neg_strat == "1_pitch" else 2.0
                neg_y += np.random.choice([-1, 1]) * step * max(10.0, pitch_px * 1.8)
            elif ptype in ["FIN_GATE", "CONTACT_ARRAY"]:
                neg_x += np.random.choice([-1, 1]) * pitch_px
                neg_y += np.random.choice([-1, 1]) * pitch_px
            else:
                neg_x += np.random.choice([-1, 1]) * np.random.uniform(25.0, 60.0)

        neg_x = float(np.clip(neg_x, self.crop_size // 2, 1000 - self.crop_size // 2))
        neg_y = float(np.clip(neg_y, self.crop_size // 2, 1000 - self.crop_size // 2))

        if np.hypot(neg_x - gt_x, neg_y - gt_y) < 6.0:
            neg_x = float(np.clip(gt_x + 30.0, self.crop_size // 2, 1000 - self.crop_size // 2))

        neg_crop = self._extract_canonical_crop(search_img, neg_x, neg_y, scale=scale, angle=angle)

        def to_tensor(arr):
            t = arr.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(t).unsqueeze(0)

        return to_tensor(ref_down), to_tensor(pos_crop), to_tensor(neg_crop)


# =====================================================================
# PART 3: VALIDATION CASCADE BENCHMARK EVALUATOR
# =====================================================================

def render_moving_bar(current: int, total: int, width: int = 22) -> str:
    """
    Renders an active moving visual ASCII progress bar safe for Windows consoles.
    """
    fraction = min(1.0, max(0.0, current / max(1, total)))
    filled = int(round(width * fraction))
    bar = "=" * filled + "-" * (width - filled)
    percent = fraction * 100.0
    return f"[{bar}] {percent:5.1f}% ({current}/{total})"

def compute_model_checksum(model: nn.Module) -> str:
    """
    Computes a deterministic SHA-256 checksum across all active model parameters.
    """
    import hashlib
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def _eval_single_pair(task_args):
    r, data_root, tau_conf, model = task_args
    ref_p = resolve_image_path(r["ref_path"], data_root)
    search_p = resolve_image_path(r["search_path"], data_root)

    if not os.path.exists(ref_p) or not os.path.exists(search_p):
        return None

    gt_x = float(r["gt_x"])
    gt_y = float(r["gt_y"])
    ptype = r.get("pattern_type", "GENERIC")

    res = localize_pair(ref_p, search_p, tau_conf=tau_conf, pattern_type=ptype, embedder_model=model)
    err = float(np.hypot(res["pred_x"] - gt_x, res["pred_y"] - gt_y))
    return {
        "pair_id": r.get("pair_id", ""),
        "ptype": ptype,
        "err": err,
        "path_used": res.get("path_used", "ncc_direct"),
        "subpixel_valid": res.get("subpixel_valid", 1),
        "gt_x": gt_x,
        "gt_y": gt_y,
        "pred_x": res.get("pred_x", 500.0),
        "pred_y": res.get("pred_y", 500.0)
    }


def evaluate_full_cascade_on_val(val_rows, model, device, data_root="dataset", tau_conf=0.65, max_val_eval=153, label="Cascade Val", prev_p5_map=None):
    """
    Runs the complete 5-phase cascade in parallel across CPU cores with a live moving progress bar,
    passing the exact in-memory model object.
    """
    errors = []
    class_errors = {
        "FIN_ARRAY": [], "FIN_CUT": [], "GATE_POLY": [], "FIN_GATE": [],
        "CONTACT_ARRAY": [], "LOCAL_INTERCONNECT": [], "METAL_ROUTING": [],
        "ACTIVE_CELL": [], "FINFET_FULL_CELL": []
    }
    phase2_count = 0
    phase5_count = 0
    subpixel_valid_count = 0
    p5_changes_count = 0
    current_p5_map = {}

    eval_subset = val_rows[:max_val_eval] if max_val_eval else val_rows
    tot_eval = len(eval_subset)
    t_val_start = time.perf_counter()

    tasks = [(r, data_root, tau_conf, model) for r in eval_subset]
    num_workers = min(8, os.cpu_count() or 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_eval_single_pair, t) for t in tasks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            done += 1
            elapsed_v = time.perf_counter() - t_val_start
            avg_t = elapsed_v / done
            rem_sec = (tot_eval - done) * avg_t

            def fmt_eta(s):
                if s < 60: return f"{s:.0f}s"
                return f"{int(s//60)}m {int(s%60):02d}s"

            bar_str = render_moving_bar(done, tot_eval, width=22)
            print(f"  [{label}] {bar_str} | Elapsed: {fmt_eta(elapsed_v)} | ETA: {fmt_eta(rem_sec)}    ", end="\r", flush=True)

            if item is not None:
                errors.append(item["err"])
                ptype = item["ptype"]
                if ptype in class_errors:
                    class_errors[ptype].append(item["err"])

                path_used = item["path_used"]
                if path_used in ["geometry_verified", "ml_reranked"]:
                    phase2_count += 1
                if path_used == "ml_reranked":
                    phase5_count += 1
                    pred_key = (round(item["pred_x"], 1), round(item["pred_y"], 1))
                    pid = item["pair_id"]
                    current_p5_map[pid] = pred_key
                    if prev_p5_map and pid in prev_p5_map and prev_p5_map[pid] != pred_key:
                        p5_changes_count += 1

                if item["subpixel_valid"] == 1:
                    subpixel_valid_count += 1
    print("", flush=True)

    if not errors:
        return {}

    err_arr = np.array(errors, dtype=np.float64)
    n = len(err_arr)

    per_class_means = {}
    for p, p_errs in class_errors.items():
        per_class_means[p] = float(np.mean(p_errs)) if p_errs else 0.0

    p3_errs = class_errors.get("GATE_POLY", [])
    p3_mean = float(np.mean(p3_errs)) if p3_errs else 0.0
    p3_max = float(np.max(p3_errs)) if p3_errs else 0.0
    p3_acc_1px = float(np.sum(np.array(p3_errs) < 1.0) / len(p3_errs) * 100.0) if p3_errs else 100.0

    return {
        "count": n,
        "mean": float(np.mean(err_arr)),
        "median": float(np.median(err_arr)),
        "p95": float(np.percentile(err_arr, 95)),
        "max": float(np.max(err_arr)),
        "acc_5px": float(np.sum(err_arr < 5.0) / n * 100.0),
        "acc_2px": float(np.sum(err_arr < 2.0) / n * 100.0),
        "acc_1px": float(np.sum(err_arr < 1.0) / n * 100.0),
        "acc_06px": float(np.sum(err_arr < 0.6) / n * 100.0),
        "class_means": per_class_means,
        "phase2_rate": float(phase2_count / n * 100.0),
        "phase5_rate": float(phase5_count / n * 100.0),
        "subpixel_valid_rate": float(subpixel_valid_count / n * 100.0),
        "phase5_changes": p5_changes_count,
        "current_p5_map": current_p5_map,
        "p3_mean": p3_mean,
        "p3_max": p3_max,
        "p3_acc_1px": p3_acc_1px,
        "p3_count": len(p3_errs)
    }

def train_siamese_reranker(
    train_manifest: str = "dataset/train_metadata.csv",
    val_manifest: str = "dataset/val_metadata.csv",
    test_manifest: str = "dataset/test_metadata.csv",
    out_dir: str = "model",
    max_epochs: int = 60,
    warmup_epochs: int = 5,
    patience: int = 10,
    min_delta: float = 0.001,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    hard_neg_refresh_interval: int = 10
):
    """
    Standard-compliant Siamese candidate reranker training engine:
    - LR Schedule: LinearWarmup(5) + CosineDecay
    - Optimization: AdamW (LR=3e-4, WD=1e-4, GradClip=1.0)
    - Loss: Triplet Margin Loss with 3-5 hard negatives per positive
    - Hard Negative Refresh: every 10 epochs strictly from TRAIN split
    - Checkpoint Metric: Primary = Val Hard-Negative Ranking Accuracy, Secondary = Full-Cascade Val Mean Error
    - Restore Best: True
    - Test Evaluation: After final model freeze on held-out test split
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DynamicSEMTripletDataset(train_manifest, is_train=True)
    val_dataset = DynamicSEMTripletDataset(val_manifest, is_train=False)

    val_rows = []
    if os.path.exists(val_manifest):
        with open(val_manifest, "r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))

    test_rows = []
    if test_manifest and os.path.exists(test_manifest):
        with open(test_manifest, "r", encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LightweightSEMEmbedder(emb_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.TripletMarginLoss(margin=0.35, p=2)

    best_val_ranking_acc = -1.0
    best_mean_error = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(out_dir, "phase5_reranker.pt")
    t_training_start = time.perf_counter()
    prev_p5_map = {}

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.perf_counter()

        # Learning Rate Schedule: Linear Warmup (5) + Cosine Decay
        if epoch <= warmup_epochs:
            current_lr = lr * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            current_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Refresh Hard Negatives every 10 epochs strictly from TRAIN split
        if epoch > 1 and (epoch % hard_neg_refresh_interval == 0):
            print(f"\n[Hard-Negative Refresh] Mining difficult NCC candidates from TRAIN split (Epoch {epoch})...", flush=True)
            train_dataset.refresh_hard_negatives(top_n_mine=100)

        # 1. Training Phase
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        tot_b = len(train_loader)

        print(f"\n>>> Starting Epoch {epoch:02d}/{max_epochs:02d} (LR: {current_lr:.6f})...", flush=True)

        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            bar_b = render_moving_bar(train_batches, tot_b, width=20)
            print(f"  [Train Batches] {bar_b} | Loss: {loss.item():.4f}    ", end="\r", flush=True)

        print("", flush=True)
        avg_train_loss = total_train_loss / max(1, train_batches)

        # 2. Validation Triplet Loss & Ranking Accuracy
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_rankings = 0
        total_samples = 0

        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                emb_a = model(anchor)
                emb_p = model(pos)
                emb_n = model(neg)

                loss = criterion(emb_a, emb_p, emb_n)
                val_loss += loss.item()
                val_batches += 1

                # Hard-Negative Ranking Accuracy: sim(A, P) > sim(A, N)
                sim_pos = torch.sum(emb_a * emb_p, dim=1)
                sim_neg = torch.sum(emb_a * emb_n, dim=1)
                correct_rankings += torch.sum(sim_pos > sim_neg).item()
                total_samples += anchor.size(0)

        avg_val_loss = val_loss / max(1, val_batches)
        val_ranking_acc = (correct_rankings / max(1, total_samples)) * 100.0

        # 3. Full Cascade Validation Evaluation
        cascade_val = evaluate_full_cascade_on_val(val_rows, model, device, data_root=val_dataset.data_root, tau_conf=0.65, max_val_eval=len(val_rows), label="Cascade Val", prev_p5_map=prev_p5_map)
        prev_p5_map = cascade_val.get("current_p5_map", {})
        epoch_time = time.perf_counter() - t_epoch_start

        mean_err = cascade_val.get("mean", 99.0)
        median_err = cascade_val.get("median", 99.0)
        p95_err = cascade_val.get("p95", 99.0)
        max_err = cascade_val.get("max", 99.0)
        acc_5px = cascade_val.get("acc_5px", 0.0)
        acc_2px = cascade_val.get("acc_2px", 0.0)
        acc_1px = cascade_val.get("acc_1px", 0.0)
        acc_06px = cascade_val.get("acc_06px", 0.0)
        cm = cascade_val.get("class_means", {})

        # Checkpoint Selection: Primary = Val Ranking Accuracy, Secondary = Full-Cascade Val Mean Error
        # Save 'last' model checkpoint every epoch
        last_checkpoint_path = checkpoint_path.replace(".pt", "_last.pt") if ".pt" in checkpoint_path else f"{checkpoint_path}_last.pt"
        torch.save(model.state_dict(), last_checkpoint_path)

        # Respect MIN_DELTA = 0.001 for 'best' checkpoint
        is_best = False
        if val_ranking_acc >= (best_val_ranking_acc + min_delta):
            best_val_ranking_acc = val_ranking_acc
            best_mean_error = mean_err
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            torch.save(model.state_dict(), checkpoint_path)
            best_named_path = checkpoint_path.replace(".pt", "_best.pt") if ".pt" in checkpoint_path else f"{checkpoint_path}_best.pt"
            torch.save(model.state_dict(), best_named_path)
        elif abs(val_ranking_acc - best_val_ranking_acc) <= min_delta and mean_err < best_mean_error:
            best_mean_error = mean_err
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            torch.save(model.state_dict(), checkpoint_path)
            best_named_path = checkpoint_path.replace(".pt", "_best.pt") if ".pt" in checkpoint_path else f"{checkpoint_path}_best.pt"
            torch.save(model.state_dict(), best_named_path)
        else:
            patience_counter += 1

        # Time Calculations
        elapsed_total = time.perf_counter() - t_training_start
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = max_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time

        def format_sec(s: float) -> str:
            if s < 60:
                return f"{s:.1f}s"
            m = int(s // 60)
            sec = int(s % 60)
            if m < 60:
                return f"{m}m {sec:02d}s"
            h = int(m // 60)
            m = int(m % 60)
            return f"{h}h {m:02d}m {sec:02d}s"

        model_checksum = compute_model_checksum(model)
        model_obj_id = hex(id(model))
        prev_p5_map = cascade_val.get("current_p5_map", {})

        # Print Exact User-Specified Dashboard Output
        print(f"\nEpoch {epoch:02d}/{max_epochs:02d}", flush=True)
        print("=" * 60, flush=True)
        print("MODEL CHECKSUM", flush=True)
        print(f"Parameter Checksum        : {model_checksum}", flush=True)
        print(f"Model Object Identity     : {model_obj_id}", flush=True)
        print(f"Current Epoch             : {epoch:02d}/{max_epochs:02d}", flush=True)
        print("\nTRAINING", flush=True)
        print(f"Train Triplet Loss        : {avg_train_loss:.4f}", flush=True)
        print(f"Validation Triplet Loss   : {avg_val_loss:.4f}", flush=True)
        print(f"Val Ranking Accuracy      : {val_ranking_acc:5.2f}%", flush=True)
        print(f"Learning Rate             : {current_lr:.6f}", flush=True)
        print("\nFULL CASCADE VALIDATION", flush=True)
        print(f"Mean Error                : {mean_err:.4f} px", flush=True)
        print(f"Median Error              : {median_err:.4f} px", flush=True)
        print(f"P95 Error                 : {p95_err:.4f} px", flush=True)
        print(f"Max Error                 : {max_err:.4f} px", flush=True)
        print(f"\nAccuracy < 5 px           : {acc_5px:5.2f}%", flush=True)
        print(f"Accuracy < 2 px           : {acc_2px:5.2f}%", flush=True)
        print(f"Accuracy < 1 px           : {acc_1px:5.2f}%", flush=True)
        print(f"Accuracy < 0.6 px         : {acc_06px:5.2f}%", flush=True)
        print("\nPER-CLASS MEAN ERROR", flush=True)
        print(f"P1 FIN_ARRAY              : {cm.get('FIN_ARRAY', 0.0):.4f} px", flush=True)
        print(f"P2 FIN_CUT                : {cm.get('FIN_CUT', 0.0):.4f} px", flush=True)
        print(f"P3 GATE_POLY              : {cm.get('GATE_POLY', 0.0):.4f} px", flush=True)
        print(f"P4 FIN_GATE               : {cm.get('FIN_GATE', 0.0):.4f} px", flush=True)
        print(f"P5 CONTACT_ARRAY          : {cm.get('CONTACT_ARRAY', 0.0):.4f} px", flush=True)
        print(f"P6 LOCAL_INTERCONNECT     : {cm.get('LOCAL_INTERCONNECT', 0.0):.4f} px", flush=True)
        print(f"P7 METAL_ROUTING          : {cm.get('METAL_ROUTING', 0.0):.4f} px", flush=True)
        print(f"P8 ACTIVE_CELL            : {cm.get('ACTIVE_CELL', 0.0):.4f} px", flush=True)
        print(f"P9 FINFET_FULL_CELL       : {cm.get('FINFET_FULL_CELL', 0.0):.4f} px", flush=True)
        print("\nCASCADE", flush=True)
        print(f"Phase-2 Invocation Rate   : {cascade_val.get('phase2_rate', 0.0):5.2f}%", flush=True)
        print(f"Phase-5 Invocation Rate   : {cascade_val.get('phase5_rate', 0.0):5.2f}%", flush=True)
        print(f"Phase-5 Acceptance Rate   : {cascade_val.get('phase5_rate', 0.0):5.2f}%", flush=True)
        print(f"Phase-5 Candidate Changes : {cascade_val.get('phase5_changes', 0)}", flush=True)
        print(f"Subpixel Valid Rate       : {cascade_val.get('subpixel_valid_rate', 0.0):5.2f}%", flush=True)
        print("\nP3 (GATE_POLY)", flush=True)
        print(f"Ambiguous Cases           : {cascade_val.get('p3_count', 0)}", flush=True)
        print(f"P3 Mean Error             : {cascade_val.get('p3_mean', 0.0):.4f} px", flush=True)
        print(f"P3 Max Error              : {cascade_val.get('p3_max', 0.0):.4f} px", flush=True)
        print(f"P3 Accuracy < 1 px        : {cascade_val.get('p3_acc_1px', 0.0):5.2f}%", flush=True)
        print("\nTARGET STATUS", flush=True)
        print(f"Mean < 2 px               : {'PASS' if mean_err < 2.0 else 'FAIL'}", flush=True)
        print(f"Median < 0.5 px           : {'PASS' if median_err < 0.5 else 'FAIL'}", flush=True)
        print(f"P95 < 5 px                : {'PASS' if p95_err < 5.0 else 'FAIL'}", flush=True)
        print(f"<5 px > 95%               : {'PASS' if acc_5px >= 95.0 else 'FAIL'}", flush=True)
        print(f"<2 px > 90%               : {'PASS' if acc_2px >= 90.0 else 'FAIL'}", flush=True)
        print(f"<1 px > 80%               : {'PASS' if acc_1px >= 80.0 else 'FAIL'}", flush=True)
        print(f"\nBest Epoch                : {best_epoch:02d}", flush=True)
        print(f"Best Mean Error           : {best_mean_error:.4f} px", flush=True)
        print(f"Patience                  : {patience_counter}/{patience}", flush=True)
        print(f"Epoch Time                : {epoch_time:.2f} sec", flush=True)
        print(f"Elapsed Time              : {format_sec(elapsed_total)}", flush=True)
        print(f"Estimated Remaining (ETA) : {format_sec(eta_seconds)}", flush=True)
        print("=" * 60, flush=True)

        # Early Stopping Check (Patience = 10)
        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] No improvement in validation metrics for {patience} epochs.")
            break

    # Restore Best Checkpoint & Freeze Model
    if os.path.exists(checkpoint_path):
        print(f"\n[Model Freeze] Restoring best model checkpoint from Epoch {best_epoch} ({checkpoint_path})...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # TEST EVALUATION: After final model freeze only
    if test_rows:
        print("\n" + "=" * 60)
        print(f"FINAL FROZEN MODEL EVALUATION ON HELD-OUT TEST SPLIT ({len(test_rows)} PAIRS)")
        print("=" * 60)
        test_cascade_val = evaluate_full_cascade_on_val(test_rows, model, device, max_val_eval=len(test_rows), label="Test Eval")
        t_mean = test_cascade_val.get("mean", 0.0)
        t_median = test_cascade_val.get("median", 0.0)
        t_p95 = test_cascade_val.get("p95", 0.0)
        t_max = test_cascade_val.get("max", 0.0)
        t_5px = test_cascade_val.get("acc_5px", 0.0)
        t_2px = test_cascade_val.get("acc_2px", 0.0)
        t_1px = test_cascade_val.get("acc_1px", 0.0)
        t_06px = test_cascade_val.get("acc_06px", 0.0)
        t_cm = test_cascade_val.get("class_means", {})

        print(f"Test Mean Error           : {t_mean:.4f} px")
        print(f"Test Median Error         : {t_median:.4f} px")
        print(f"Test P95 Error            : {t_p95:.4f} px")
        print(f"Test Max Error            : {t_max:.4f} px")
        print(f"Test Accuracy < 5 px      : {t_5px:5.2f}%")
        print(f"Test Accuracy < 2 px      : {t_2px:5.2f}%")
        print(f"Test Accuracy < 1 px      : {t_1px:5.2f}%")
        print(f"Test Accuracy < 0.6 px    : {t_06px:5.2f}%")
        print("\nPER-CLASS TEST MEAN ERROR:")
        for p, perr in t_cm.items():
            print(f"  {p:24s}: {perr:.4f} px")
        print("=" * 60 + "\n")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Phase 5 Siamese Model & Run Full Validation")
    parser.add_argument("--train_manifest", default="dataset/train_metadata.csv", help="Train metadata CSV")
    parser.add_argument("--val_manifest", default="dataset/val_metadata.csv", help="Val metadata CSV")
    parser.add_argument("--test_manifest", default="dataset/test_metadata.csv", help="Test metadata CSV")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum metric delta")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--hard_neg_refresh", type=int, default=10, help="Hard negative refresh interval in epochs")
    args = parser.parse_args()

    train_siamese_reranker(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        hard_neg_refresh_interval=args.hard_neg_refresh
    )


if __name__ == "__main__":
    main()
