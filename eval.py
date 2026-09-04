import os
import argparse
import yaml
import torch

from data.dataset import get_cross_modal_dataloaders
from models.vit_reid import ViTCrossModalReID
from utils.metrics import evaluate_cross_modal
from train import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Cross-Modality Person Re-ID Evaluation")
    parser.add_argument("--weights", type=str, default="./runs/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to config yaml")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[Error] Checkpoint weights not found at: {args.weights}")
        print("Please train a model first using 'python train.py' or specify a valid checkpoint path via --weights.")
        return

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Loading checkpoint: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)

    # Prepare DataLoaders
    _, eval_loaders, num_classes = get_cross_modal_dataloaders(
        root_dir=cfg["dataset"]["root_dir"],
        batch_size=cfg["training"]["batch_size"],
        img_size=(cfg["dataset"]["img_height"], cfg["dataset"]["img_width"]),
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
        persistent_workers=False
    )

    # Initialize Model & Load Weights
    model = ViTCrossModalReID(
        num_classes=num_classes,
        backbone_name=cfg["model"]["backbone"],
        embed_dim=cfg["model"]["embed_dim"],
        drop_rate=cfg["model"]["drop_rate"],
        grad_checkpointing=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("[Evaluation] Successfully loaded model weights. Starting evaluation...\n")

    eval_results = evaluate_model(model, eval_loaders, device)

    print("=" * 80)
    print(" CROSS-MODALITY PERSON RE-IDENTIFICATION EVALUATION REPORT")
    print("=" * 80)
    print(f"{'Direction':<25} | {'Rank-1':<10} | {'Rank-5':<10} | {'Rank-10':<10} | {'mAP':<10}")
    print("-" * 80)
    
    for direction, metrics in eval_results.items():
        dir_label = "RGB -> Synthetic IR" if direction == "rgb_to_ir" else "Synthetic IR -> RGB"
        print(
            f"{dir_label:<25} | "
            f"{metrics['rank1']:>8.2f}% | "
            f"{metrics['rank5']:>8.2f}% | "
            f"{metrics['rank10']:>8.2f}% | "
            f"{metrics['mAP']:>8.2f}%"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
