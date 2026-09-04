import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter


def get_peak_vram_mb() -> float:
    """Returns peak GPU memory allocated in MB if CUDA is available."""
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return float(vram_mb)
    return 0.0


def reset_vram_stats():
    """Resets peak memory statistics for CUDA GPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class MetricLogger:
    """
    Logger for metrics, loss progress, VRAM memory usage, and TensorBoard summaries.
    """
    def __init__(self, log_dir: str = "./runs/exp1"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        
        # Initialize CSV header if not present
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss_total", "loss_ce", "loss_triplet", "vram_mb", "rank1", "rank5", "rank10", "mAP"])

    def log_epoch(self, epoch: int, loss_dict: dict, metrics: dict = None):
        vram_mb = get_peak_vram_mb()
        
        # Log to TensorBoard
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Loss/{k}", v, epoch)
        self.writer.add_scalar("Hardware/Peak_VRAM_MB", vram_mb, epoch)
        
        r1, r5, r10, map_val = 0.0, 0.0, 0.0, 0.0
        if metrics:
            r1 = metrics.get("rank1", 0.0)
            r5 = metrics.get("rank5", 0.0)
            r10 = metrics.get("rank10", 0.0)
            map_val = metrics.get("mAP", 0.0)
            
            self.writer.add_scalar("Metrics/Rank1", r1, epoch)
            self.writer.add_scalar("Metrics/Rank5", r5, epoch)
            self.writer.add_scalar("Metrics/Rank10", r10, epoch)
            self.writer.add_scalar("Metrics/mAP", map_val, epoch)

        # Log to CSV file
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                loss_dict.get("loss_total", 0.0),
                loss_dict.get("loss_ce", 0.0),
                loss_dict.get("loss_triplet", 0.0),
                vram_mb,
                r1, r5, r10, map_val
            ])

        print(
            f"Epoch [{epoch:02d}] | Loss: {loss_dict.get('loss_total', 0.0):.4f} | "
            f"Peak VRAM: {vram_mb:.2f} MB | Rank-1: {r1:.2f}% | mAP: {map_val:.2f}%"
        )

    def close(self):
        self.writer.close()
