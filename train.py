import os
import sys
import re
import glob
import time
import argparse
import yaml
import torch
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from data.dataset import get_cross_modal_dataloaders
from models.vit_reid import ViTCrossModalReID
from models.losses import HybridCrossModalLoss
from utils.metrics import evaluate_cross_modal
from utils.logger import MetricLogger, get_peak_vram_mb, reset_vram_stats


def print_dataset_banner(dataset_name: str = "sysu_mm01"):
    if dataset_name == "sysu_mm01":
        banner = """
    ================================================================================
    [INFO] REAL SYSU-MM01 RGB-INFRARED DATASET IN USE
    --------------------------------------------------------------------------------
    Training with real multi-camera RGB (cam1, cam2, cam4, cam5) and 
    real infrared thermal (cam3, cam6) images from SYSU-MM01.
    ================================================================================
        """
    else:
        banner = """
    ================================================================================
    [WARNING] SYNTHETIC INFRARED MODALITY IN USE (BASELINE MODE)
    --------------------------------------------------------------------------------
    This baseline uses synthetic infrared (IR) generated from Market-1501 RGB images
    (Grayscale -> CLAHE -> INFERNO Colormap).
    ================================================================================
        """
    print(banner)


def load_config(config_path: str = "./configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(save_dir: str) -> tuple:
    """
    Finds the checkpoint with the highest epoch number in save_dir.
    Returns tuple of (checkpoint_path, epoch_num).
    """
    if not os.path.exists(save_dir):
        return None, 0
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, 0
    
    latest_ckpt = None
    max_epoch = -1
    epoch_regex = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    
    for ckpt in ckpts:
        match = epoch_regex.search(os.path.basename(ckpt))
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = ckpt
                
    return latest_ckpt, max_epoch


def save_checkpoint(state: dict, save_dir: str, epoch: int, is_best: bool = False, keep_last_n: int = 3):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved epoch {epoch} checkpoint to {ckpt_path}")

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"[Checkpoint] Saved new best model to {best_path}")

    # Remove old checkpoints keeping only last N
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    
    def extract_epoch(path):
        m = re.search(r"checkpoint_epoch_(\d+)\.pth", os.path.basename(path))
        return int(m.group(1)) if m else 0

    ckpts = sorted(glob.glob(pattern), key=extract_epoch)
    if len(ckpts) > keep_last_n:
        for old_ckpt in ckpts[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
            except OSError:
                pass


def run_dry_run(model, loss_fn, dataloader, device, use_amp: bool = True):
    print("\n[Dry Run Probe] Running 1 mini-batch forward + backward pass to measure peak VRAM...")
    reset_vram_stats()
    model.train()
    
    batch = next(iter(dataloader))
    rgb_imgs = batch["rgb"].to(device)
    ir_imgs = batch["ir"].to(device)
    pids = batch["pid"].to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    try:
        scaler = GradScaler(device=dev_type, enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)
    
    start_time = time.time()
    optimizer.zero_grad()
    
    with autocast(device_type=dev_type, enabled=use_amp):
        rgb_feat, rgb_logits = model(rgb_imgs)
        ir_feat, ir_logits = model(ir_imgs)
        loss, _ = loss_fn(rgb_feat, rgb_logits, ir_feat, ir_logits, pids)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    elapsed = time.time() - start_time
    peak_vram = get_peak_vram_mb()
    
    print("\n" + "="*60)
    print(f" DRY RUN SUCCESSFUL")
    print(f" Batch Size: {rgb_imgs.size(0)} RGB + {ir_imgs.size(0)} IR (Total {rgb_imgs.size(0)*2} images)")
    print(f" Mixed Precision AMP: {'ENABLED' if use_amp else 'DISABLED'}")
    print(f" Iteration Time: {elapsed:.3f} seconds")
    print(f" Peak GPU VRAM Consumption: {peak_vram:.2f} MB / 6144.00 MB")
    print("="*60 + "\n")
    
    if peak_vram > 5500.0:
        print("[WARNING] Peak VRAM is close to the 6GB limit! Ensure gradient_checkpointing: true in config.")


def train_epoch(model, loss_fn, train_loader, optimizer, scaler, scheduler, device, accum_steps: int = 2, use_amp: bool = True):
    model.train()
    total_loss_accum = 0.0
    ce_loss_accum = 0.0
    triplet_loss_accum = 0.0
    
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        rgb_imgs = batch["rgb"].to(device, non_blocking=True)
        ir_imgs = batch["ir"].to(device, non_blocking=True)
        pids = batch["pid"].to(device, non_blocking=True)
        
        with autocast(device_type=dev_type, enabled=use_amp):
            rgb_feat, rgb_logits = model(rgb_imgs)
            ir_feat, ir_logits = model(ir_imgs)
            loss, loss_components = loss_fn(rgb_feat, rgb_logits, ir_feat, ir_logits, pids)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss_accum += loss_components["loss_total"]
        ce_loss_accum += loss_components["loss_ce"]
        triplet_loss_accum += loss_components["loss_triplet"]

    scheduler.step()
    num_batches = len(train_loader)
    return {
        "loss_total": total_loss_accum / num_batches,
        "loss_ce": ce_loss_accum / num_batches,
        "loss_triplet": triplet_loss_accum / num_batches
    }


@torch.no_grad()
def evaluate_model(model, eval_loaders, device):
    model.eval()
    results = {}
    
    for eval_key, loader_pair in eval_loaders.items():
        q_loader, g_loader = loader_pair["query"], loader_pair["gallery"]
        
        # Extract Query Features
        q_feats, q_pids, q_cams = [], [], []
        for batch in q_loader:
            imgs = batch["img"].to(device)
            feats = model.extract_features(imgs)
            q_feats.append(feats.cpu())
            q_pids.append(batch["pid"])
            q_cams.append(batch["camid"])
        q_feats = torch.cat(q_feats, dim=0)
        q_pids = torch.cat(q_pids, dim=0)
        q_cams = torch.cat(q_cams, dim=0)

        # Extract Gallery Features
        g_feats, g_pids, g_cams = [], [], []
        for batch in g_loader:
            imgs = batch["img"].to(device)
            feats = model.extract_features(imgs)
            g_feats.append(feats.cpu())
            g_pids.append(batch["pid"])
            g_cams.append(batch["camid"])
        g_feats = torch.cat(g_feats, dim=0)
        g_pids = torch.cat(g_pids, dim=0)
        g_cams = torch.cat(g_cams, dim=0)

        metrics = evaluate_cross_modal(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams)
        results[eval_key] = metrics
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Modality Person Re-ID Training Entrypoint")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--dry-run", action="store_true", help="Probe VRAM consumption for 1 batch and exit")
    parser.add_argument("--save-dir", type=str, default=None, help="Override checkpoint save directory")
    parser.add_argument("--no-resume", action="store_true", help="Force starting fresh from epoch 1 ignoring existing checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = cfg["dataset"].get("name", "sysu_mm01")
    print_dataset_banner(dataset_name)
    
    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Determine Save Directory (Colab Google Drive support vs Config vs CLI)
    colab_drive_dir = "/content/drive/MyDrive/CrossModality_ReID/runs"
    if args.save_dir:
        save_dir = args.save_dir
    elif os.path.exists("/content/drive/MyDrive/CrossModality_ReID"):
        save_dir = colab_drive_dir
        print(f"[Checkpoint] Google Drive detected. Using save directory: {save_dir}")
    else:
        save_dir = cfg["logging"].get("save_dir", "./runs")
        
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Checkpoint] Checkpoint directory ready: {save_dir}")

    # Automated Dataset Loading (REAL SYSU-MM01 or Synthetic Baseline)
    dataset_name = cfg["dataset"].get("name", "sysu_mm01")
    subset_size = cfg["dataset"].get("subset_size", 2000)

    train_loader, eval_loaders, num_classes = get_cross_modal_dataloaders(
        root_dir=cfg["dataset"].get("root_dir"),
        batch_size=cfg["training"]["batch_size"],
        img_size=(cfg["dataset"]["img_height"], cfg["dataset"]["img_width"]),
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
        persistent_workers=cfg["system"]["persistent_workers"],
        dataset_name=dataset_name,
        subset_size=subset_size
    )
    print(f"[Dataset] Train DataLoader ready ({len(train_loader.dataset)} image pairs, {num_classes} identities)")

    # Model & Loss Function
    model = ViTCrossModalReID(
        num_classes=num_classes,
        backbone_name=cfg["model"]["backbone"],
        embed_dim=cfg["model"]["embed_dim"],
        drop_rate=cfg["model"]["drop_rate"],
        grad_checkpointing=cfg["model"]["grad_checkpointing"]
    ).to(device)

    loss_fn = HybridCrossModalLoss(
        num_classes=num_classes,
        lambda_ce=cfg["loss"]["lambda_ce"],
        lambda_triplet=cfg["loss"]["lambda_triplet"],
        margin=cfg["loss"]["margin"],
        epsilon=cfg["loss"]["label_smoothing"]
    ).to(device)

    if args.dry_run:
        run_dry_run(model, loss_fn, train_loader, device, use_amp=cfg["training"]["use_amp"])
        sys.exit(0)

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6
    )
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    try:
        scaler = GradScaler(device=dev_type, enabled=cfg["training"]["use_amp"])
    except TypeError:
        scaler = GradScaler(enabled=cfg["training"]["use_amp"])

    # Resume Checkpoint Logic
    start_epoch = 1
    best_rank1 = 0.0

    if not args.no_resume:
        latest_ckpt_path, latest_epoch = find_latest_checkpoint(save_dir)
        if latest_ckpt_path and os.path.exists(latest_ckpt_path):
            print(f"[Checkpoint] Resuming from epoch {latest_epoch}")
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    print(f"[Checkpoint] Warning: Could not load optimizer state: {e}")
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    print(f"[Checkpoint] Warning: Could not load scheduler state: {e}")
            if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception as e:
                    print(f"[Checkpoint] Warning: Could not load scaler state: {e}")
                    
            start_epoch = checkpoint.get("epoch", latest_epoch) + 1
            best_rank1 = checkpoint.get("best_rank1", 0.0)
            print(f"[Checkpoint] Starting from epoch {start_epoch}")
        else:
            print(f"[Checkpoint] Starting from epoch 1")
    else:
        print("[Checkpoint] Starting fresh training from epoch 1 (--no-resume flag passed)")

    # Logger Setup
    logger = MetricLogger(log_dir=save_dir)

    print(f"\n[Training Loop] Starting Cross-Modality Re-ID Training (Epochs {start_epoch} -> {cfg['training']['epochs']})...")
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        reset_vram_stats()
        loss_metrics = train_epoch(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            accum_steps=cfg["training"]["grad_accum_steps"],
            use_amp=cfg["training"]["use_amp"]
        )

        # Evaluate every checkpoint_interval
        eval_metrics = None
        if epoch % cfg["logging"]["checkpoint_interval"] == 0 or epoch == cfg["training"]["epochs"]:
            print(f"[Evaluation] Epoch {epoch}: Running Cross-Modality Matching Evaluation...")
            eval_results = evaluate_model(model, eval_loaders, device)
            eval_metrics = eval_results["rgb_to_ir"]
            print(
                f"[Evaluation Result (RGB->IR)] Rank-1: {eval_metrics['rank1']:.2f}% | "
                f"Rank-5: {eval_metrics['rank5']:.2f}% | Rank-10: {eval_metrics['rank10']:.2f}% | "
                f"mAP: {eval_metrics['mAP']:.2f}%"
            )

        # Log epoch metrics and VRAM
        logger.log_epoch(epoch, loss_metrics, eval_metrics)

        # Save Checkpoint
        is_best = False
        if eval_metrics and eval_metrics["rank1"] > best_rank1:
            best_rank1 = eval_metrics["rank1"]
            is_best = True

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_rank1": best_rank1,
                "config": cfg
            },
            save_dir=save_dir,
            epoch=epoch,
            is_best=is_best,
            keep_last_n=cfg["logging"]["keep_last_n"]
        )

    logger.close()
    print("\n[Training Finished] All epochs completed successfully.")


if __name__ == "__main__":
    main()
