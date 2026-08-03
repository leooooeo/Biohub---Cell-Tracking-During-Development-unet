#!/usr/bin/env python
"""One-shot: pre-cache frozen-UNet node features, then train ONLY the transformer.

This is the fast path for iterating on the edge/link transformer (the
``t+1-as-query`` reorientation + division head) without paying for the 3D UNet
forward/backward every epoch.

Two phases, run back-to-back by default:

1. **cache** — load the shipped full model (``edge_predictor_best.pth``), freeze
   the UNet + detection head, and run the exact training-time
   ``encode → detect → match → index`` pipeline **once** over every consecutive
   frame pair.  For each pair it stores the transformer's inputs (per-node
   feature = UNet feature ⊕ positional embedding, original-resolution coords)
   and the matched edge target (as a sparse index list).  Written as one
   ``.pt`` shard per video under ``--cache-dir``.

2. **train** — instantiate a fresh ``SimpleNodeTransformer`` (matching the arch
   inside ``UNetNodeTransformer``), train it on the cached pairs with the same
   edge + division losses, and save a full checkpoint (frozen UNet + detect head
   copied from the shipped weights, transformer replaced by the trained one) that
   ``predict_unet_transformer.py`` can load unchanged.

Example (Colab)::

    export BIOHUB_DATA_DIR=/root/.cache/kagglehub/competitions/\
biohub-cell-tracking-during-development/train
    python scripts/cache_and_train_transformer.py --epochs 80

Caching is deterministic (no image augmentation), so it is done once and reused.
Re-run with ``--phase train`` to iterate on the transformer without re-caching.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Make sibling scripts and the src package importable without an install step.
# This file lives at repo/scripts/; the package is at repo/src/biohub_tracking.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_unet_transformer import (  # noqa: E402
    DIV_LOSS_WEIGHT,
    UNetNodeTransformer,
    FrameWindowDataset,
    build_matched_edge_targets,
    compute_batch_division_loss,
    compute_batch_loss,
    detect_and_match,
    load_dataset_windows,
    _POS_EMBED_DIM,
    _evaluate_pair,
)
from biohub_tracking.models import SimpleNodeTransformer, TemporalUNet3D  # noqa: E402
from dataspec import DATASET_PATH, WEIGHTS_PATH  # noqa: E402


# =============================================================================
# Splits
# =============================================================================

def resolve_splits(
    data_dir: Path,
    splits_file: Path | None,
    fold: int,
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return (train_names, val_names).

    If *splits_file* exists it is used (``folds[fold]['train'/'test']``).
    Otherwise an auto split is built from every ``*.geff`` next to a ``*.zarr``
    in *data_dir* (training needs GT tracks).
    """
    if splits_file is not None and splits_file.exists():
        folds = json.loads(splits_file.read_text())
        return folds[fold]["train"], folds[fold]["test"]

    names = sorted(
        p.stem for p in data_dir.glob("*.geff")
        if (data_dir / f"{p.stem}.zarr").exists()
    )
    if not names:
        raise FileNotFoundError(
            f"No '<name>.zarr' + '<name>.geff' pairs found in {data_dir}. "
            f"Set --data-dir / $BIOHUB_DATA_DIR to the folder holding them."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    n_val = max(1, int(round(len(names) * val_frac)))
    val_idx = set(perm[:n_val].tolist())
    train = [names[i] for i in range(len(names)) if i not in val_idx]
    val = [names[i] for i in range(len(names)) if i in val_idx]
    print(f"Auto split ({len(names)} datasets): {len(train)} train / {len(val)} val", flush=True)
    return train, val


# =============================================================================
# Phase 1: caching
# =============================================================================

def load_frozen_encoder(
    weights_path: Path, config: dict, device: torch.device,
) -> UNetNodeTransformer:
    """Build UNetNodeTransformer and load UNet + detect head from shipped weights.

    The transformer sub-module is left as-is (it is not used for caching); only
    ``unet.*`` and ``detect_head.*`` need to match, so ``strict=False`` is fine.
    """
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    enc_missing = [k for k in missing if k.startswith(("unet.", "detect_head."))]
    if enc_missing:
        raise RuntimeError(f"Shipped weights missing encoder keys: {enc_missing[:5]} ...")
    print(f"Loaded encoder (UNet+detect). transformer keys re-init: "
          f"{len([k for k in missing if k.startswith('transformer.')])} missing", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def cache_split(
    model: UNetNodeTransformer,
    names: list[str],
    data_dir: Path,
    cache_dir: Path,
    config: dict,
    device: torch.device,
    pool_kernel_um: float,
    det_threshold: float,
    max_frames: int | None,
    force: bool,
) -> int:
    """Cache transformer inputs for every consecutive pair; one .pt per video."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    downsample = tuple(config["downsample"])
    window_size = config["window_size"]
    n_pairs_total = 0

    for name in tqdm(names, desc=f"cache[{cache_dir.name}]"):
        shard_path = cache_dir / f"{name}.pt"
        if shard_path.exists() and not force:
            n_pairs_total += len(torch.load(shard_path, weights_only=False))
            continue

        video_meta, windows = load_dataset_windows(
            data_dir / name, window_size=window_size,
            max_frames=max_frames, downsample=downsample,
        )
        if not windows:
            continue

        ds = FrameWindowDataset([(video_meta, windows)], augmentations=None)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        pairs: list[dict] = []
        for batch in loader:
            imgs = batch["imgs"].to(device, dtype=torch.float32)   # (1, W, *sp)
            coords = batch["coords"].to(device)                    # (1, W, M, 3)
            masks = batch["masks"].to(device)                      # (1, W, M)
            targets = batch["targets"].to(device)                  # (1, W-1, M, M)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())
            ds_scale = batch["downsample"][0].to(device)           # (3,)
            W = imgs.shape[1]

            unet_out, det_logits = model.encode(imgs)

            frame_det = []
            for i in range(W):
                det_c, det_p, det_m, matches = detect_and_match(
                    det_logits[i], coords[:, i], masks[:, i],
                    image_shape, voxel_size=voxel_size,
                    pool_kernel_um=pool_kernel_um, det_threshold=det_threshold,
                    frame_index=i, window_size=W,
                )
                unet_feat = model._index_features(unet_out[:, i], det_c, det_m)
                frame_det.append((det_c, det_p, det_m, matches, unet_feat))

            for i in range(W - 1):
                ns = frame_det[i][0].shape[1]
                nt = frame_det[i + 1][0].shape[1]
                pair_target = build_matched_edge_targets(
                    frame_det[i][3], frame_det[i + 1][3], targets[:, i], ns, nt,
                )  # (1, nt, ns) = (N_t1, N_t)

                m_t = frame_det[i][2][0]       # (ns,) mothers mask
                m_t1 = frame_det[i + 1][2][0]  # (nt,) children mask
                n_t = int(m_t.sum().item())
                n_t1 = int(m_t1.sum().item())
                if n_t == 0 or n_t1 == 0:
                    continue

                # feat = UNet feature ⊕ positional embedding (transformer input).
                feat_t = torch.cat(
                    [frame_det[i][4][0, :n_t], frame_det[i][1][0, :n_t]], dim=-1)
                feat_t1 = torch.cat(
                    [frame_det[i + 1][4][0, :n_t1], frame_det[i + 1][1][0, :n_t1]], dim=-1)
                coords_t = frame_det[i][0][0, :n_t] * ds_scale       # original res
                coords_t1 = frame_det[i + 1][0][0, :n_t1] * ds_scale
                tgt = pair_target[0, :n_t1, :n_t]                    # (n_t1, n_t)
                edges = torch.nonzero(tgt > 0.5, as_tuple=False).to(torch.int32)  # (E, 2)

                pairs.append({
                    "feat_t": feat_t.half().cpu(),
                    "feat_t1": feat_t1.half().cpu(),
                    "coords_t": coords_t.half().cpu(),
                    "coords_t1": coords_t1.half().cpu(),
                    "n_t": n_t,
                    "n_t1": n_t1,
                    "edges": edges.cpu(),  # (child_row, mother_col)
                })

            del unet_out, det_logits

        torch.save(pairs, shard_path)
        n_pairs_total += len(pairs)

    return n_pairs_total


# =============================================================================
# Phase 2: cached-pair dataset + transformer-only training
# =============================================================================

class CachedPairDataset(Dataset):
    """Loads cached per-pair transformer inputs from .pt shards."""

    def __init__(self, cache_dir: Path):
        self.pairs: list[dict] = []
        for shard in sorted(cache_dir.glob("*.pt")):
            self.pairs.extend(torch.load(shard, weights_only=False))
        if not self.pairs:
            raise FileNotFoundError(f"No cached pairs in {cache_dir}. Run the cache phase first.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


def collate_pairs(batch: list[dict]) -> dict:
    """Pad a batch of cached pairs to (B, max_n, ...) with masks + dense targets."""
    B = len(batch)
    max_t = max(p["n_t"] for p in batch)
    max_t1 = max(p["n_t1"] for p in batch)
    D = batch[0]["feat_t"].shape[1]

    feat_t = torch.zeros(B, max_t, D)
    feat_t1 = torch.zeros(B, max_t1, D)
    coords_t = torch.zeros(B, max_t, 3)
    coords_t1 = torch.zeros(B, max_t1, 3)
    mask_t = torch.zeros(B, max_t, dtype=torch.bool)
    mask_t1 = torch.zeros(B, max_t1, dtype=torch.bool)
    target = torch.zeros(B, max_t1, max_t)

    for b, p in enumerate(batch):
        nt, nt1 = p["n_t"], p["n_t1"]
        feat_t[b, :nt] = p["feat_t"].float()
        feat_t1[b, :nt1] = p["feat_t1"].float()
        coords_t[b, :nt] = p["coords_t"].float()
        coords_t1[b, :nt1] = p["coords_t1"].float()
        mask_t[b, :nt] = True
        mask_t1[b, :nt1] = True
        e = p["edges"]
        if e.numel():
            target[b, e[:, 0].long(), e[:, 1].long()] = 1.0

    return {
        "feat_t": feat_t, "feat_t1": feat_t1,
        "coords_t": coords_t, "coords_t1": coords_t1,
        "mask_t": mask_t, "mask_t1": mask_t1,
        "target": target,
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_transformer(
    train_cache: Path,
    val_cache: Path,
    config: dict,
    out_dir: Path,
    shipped_weights: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    div_loss_weight: float,
    seed: int,
    use_amp: bool = True,
    use_checkpoint: bool = True,
) -> None:
    torch.manual_seed(seed)
    train_ds = CachedPairDataset(train_cache)
    val_ds = CachedPairDataset(val_cache)
    print(f"Cached pairs: {len(train_ds)} train / {len(val_ds)} val", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_pairs,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_pairs,
        persistent_workers=num_workers > 0,
    )

    # Fresh transformer, identical arch to the one inside UNetNodeTransformer.
    transformer = SimpleNodeTransformer(
        feat_dim=config["unet_out_channels"] + 4 * _POS_EMBED_DIM,
        hidden_dim=128, n_heads=4, n_blocks=4, dropout=0.3,
        use_checkpoint=use_checkpoint,
    ).to(device)
    n_params = sum(p.numel() for p in transformer.parameters())
    amp_on = use_amp and device.type == "cuda"
    print(f"Transformer params: {n_params:,} | amp={amp_on} | grad_checkpoint={use_checkpoint}", flush=True)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)

    # Encoder weights to splice into the saved checkpoint (kept on CPU).
    shipped = torch.load(shipped_weights, map_location="cpu", weights_only=True)
    enc_state = {k: v for k, v in shipped.items()
                 if k.startswith(("unet.", "detect_head."))}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {k: config[k] for k in ("unet_out_channels", "unet_layers", "downsample",
                                "window_size", "pool_kernel_um")}, indent=2))
    save_path = out_dir / "edge_predictor_best.pth"
    best_score = -1.0

    for epoch in range(epochs):
        t0 = time.monotonic()
        transformer.train()
        total, n = 0.0, 0
        for batch in train_loader:
            b = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                edge_logits, div_logits = transformer(
                    b["feat_t"], b["feat_t1"], b["coords_t"], b["coords_t1"],
                    b["mask_t"], b["mask_t1"],
                )
            # Losses in fp32 (binary_cross_entropy is autocast-unsafe).
            edge_logits = edge_logits.float()
            div_logits = div_logits.float()
            edge_loss = compute_batch_loss(edge_logits, b["target"], b["mask_t"], b["mask_t1"])
            div_loss = compute_batch_division_loss(div_logits, b["target"], b["mask_t"], b["mask_t1"])
            loss = edge_loss + div_loss_weight * div_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * b["feat_t"].shape[0]
            n += b["feat_t"].shape[0]
        train_loss = total / max(n, 1)

        # Validation: positive-edge precision/recall/F1 (meaningful under the
        # extreme class imbalance) plus the legacy pairwise accuracy.
        transformer.eval()
        correct, tot = 0, 0
        tp = fp = fn = 0
        with torch.no_grad():
            for batch in val_loader:
                b = _to_device(batch, device)
                with torch.autocast(device_type=device.type, enabled=amp_on):
                    edge_logits, _ = transformer(
                        b["feat_t"], b["feat_t1"], b["coords_t"], b["coords_t1"],
                        b["mask_t"], b["mask_t1"],
                    )
                edge_logits = edge_logits.float()
                for i in range(b["feat_t"].shape[0]):
                    nt = int(b["mask_t"][i].sum()); nt1 = int(b["mask_t1"][i].sum())
                    lg = edge_logits[i, :nt1, :nt]
                    tg = b["target"][i, :nt1, :nt]
                    _, c, t = _evaluate_pair(lg, tg)
                    correct += c; tot += t
                    # Positive-edge stats over the annotated submatrix.
                    active = (tg.sum(dim=1, keepdim=True) > 0) | (tg.sum(dim=0, keepdim=True) > 0)
                    pred = (torch.softmax(lg, dim=1) > 0.5) & active
                    gt = tg > 0.5
                    tp += int((pred & gt).sum())
                    fp += int((pred & ~gt).sum())
                    fn += int((~pred & gt).sum())
        val_acc = correct / max(tot, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        is_best = f1 >= best_score
        if is_best:
            best_score = f1
            full = {**enc_state,
                    **{f"transformer.{k}": v for k, v in transformer.state_dict().items()}}
            torch.save(full, save_path)
        print(f"  Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_F1={f1:.4f} (P={prec:.3f} R={rec:.3f}) | val_acc={val_acc:.4f} | "
              f"bestF1={best_score:.4f} {'*' if is_best else ' '} | "
              f"{time.monotonic()-t0:.1f}s", flush=True)

    print(f"\nBest val_F1={best_score:.4f}, saved full checkpoint to {save_path}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["cache", "train", "all", "splits"], default="all",
                   help="'splits' only writes the auto train/val split JSON and exits "
                        "(no caching, no training) so predict can evaluate on the same val set.")
    p.add_argument("--data-dir", type=str, default=None, help="Default: $BIOHUB_DATA_DIR / dataspec")
    p.add_argument("--splits", type=str, default=None, help="Optional dataset_splits.json")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1, help="Auto-split validation fraction")
    p.add_argument("--shipped-weights", type=str, default=None,
                   help="Full model weights providing the frozen UNet+detect head. "
                        "Default: weights/unet_transformer/split_0/edge_predictor_best.pth")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Default: <repo>/cache/unet_feats")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to save the trained checkpoint. "
                        "Default: weights/unet_transformer_frozen/split_{fold}")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Workers for the training loader. Cached pairs live in RAM, so 0 "
                        "(no fork-copy) is usually best; caching I/O uses a fixed 2 internally.")
    p.add_argument("--div-loss-weight", type=float, default=DIV_LOSS_WEIGHT)
    p.add_argument("--pool-kernel-um", type=float, default=5.0)
    p.add_argument("--det-threshold", type=float, default=0.3,
                   help="Detection logit threshold at cache time. Higher = fewer "
                        "candidate nodes = much smaller N² pair matrices = faster "
                        "(at some cost to negative/GT-match coverage). Default 0.3.")
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                   help="Disable mixed-precision (AMP) training.")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false", default=True,
                   help="Disable gradient checkpointing in the transformer (faster but "
                        "much higher memory). On by default — keep it on for 16 GB GPUs "
                        "like the Colab T4; only turn off on large-VRAM GPUs.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per video (smoke test)")
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--force-recache", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else None

    # The shipped weights may sit under repo/weights (dataspec) or at the git
    # root's weights/ (the checked-in layout). Search both unless overridden.
    if args.shipped_weights:
        shipped_weights = Path(args.shipped_weights)
    else:
        rel = Path("unet_transformer/split_0/edge_predictor_best.pth")
        roots = [WEIGHTS_PATH, repo_root.parent / "weights"]
        shipped_weights = next(
            (r / rel for r in roots if (r / rel).exists()), roots[0] / rel)
    cache_root = Path(args.cache_dir) if args.cache_dir else (repo_root / "cache" / "unet_feats")
    out_dir = Path(args.out_dir) if args.out_dir else (
        WEIGHTS_PATH / "unet_transformer_frozen" / f"split_{args.fold}")

    # Architecture config comes from the shipped weights' config.json.
    cfg_path = shipped_weights.parent / "config.json"
    config = {"unet_out_channels": 32, "unet_layers": [32, 64, 128],
              "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": args.pool_kernel_um}
    if cfg_path.exists():
        config.update(json.loads(cfg_path.read_text()))
    config["pool_kernel_um"] = args.pool_kernel_um

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"data_dir={data_dir}\nshipped_weights={shipped_weights}\n"
          f"cache_dir={cache_root}\nout_dir={out_dir}\ndevice={device}", flush=True)

    train_names, val_names = resolve_splits(
        data_dir, splits_file, args.fold, args.val_frac, args.seed)

    # Persist the (auto) split so predict_unet_transformer.py can evaluate on the
    # exact same val videos: pass --splits <this file> --split 0 to that script.
    if splits_file is None:
        cache_root.mkdir(parents=True, exist_ok=True)
        auto_splits = cache_root / "dataset_splits_auto.json"
        auto_splits.write_text(json.dumps(
            [{"train": train_names, "test": val_names}], indent=2))
        print(f"Wrote auto split ({len(train_names)}/{len(val_names)}) to {auto_splits}\n"
              f"  -> evaluate with: predict_unet_transformer.py --splits {auto_splits} --split 0 --evaluate",
              flush=True)
    if args.phase == "splits":
        return

    train_cache = cache_root / "train"
    val_cache = cache_root / "val"

    if args.phase in ("cache", "all"):
        model = load_frozen_encoder(shipped_weights, config, device)
        n_tr = cache_split(model, train_names, data_dir, train_cache, config, device,
                           args.pool_kernel_um, args.det_threshold, args.max_frames, args.force_recache)
        n_va = cache_split(model, val_names, data_dir, val_cache, config, device,
                           args.pool_kernel_um, args.det_threshold, args.max_frames, args.force_recache)
        print(f"Cached {n_tr} train + {n_va} val pairs into {cache_root}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.phase in ("train", "all"):
        train_transformer(
            train_cache, val_cache, config, out_dir, shipped_weights, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            num_workers=args.num_workers, div_loss_weight=args.div_loss_weight,
            seed=args.seed, use_amp=args.amp, use_checkpoint=args.grad_checkpoint,
        )


if __name__ == "__main__":
    main()
