import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    Formula: y_smoothed = (1 - epsilon) * y + epsilon / num_classes
    """
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smoothed * log_probs).mean(0).sum()
        return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss for Cross-Modality Feature Embeddings.
    For each anchor, mines the hardest positive and hardest negative in the mini-batch.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Feature embeddings [B, D]
            targets: Person IDs [B]
        """
        n = inputs.size(0)
        # Compute pairwise Euclidean distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # Pairwise distance matrix

        # Mask for identity matching
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        # For each sample, find hardest positive (max distance) and hardest negative (min distance)
        dist_ap, _ = torch.max(dist * is_pos.float(), dim=1)
        
        # Replace zero distances in neg mask with infinity to find true minimum
        dist_an, _ = torch.min(dist + 1e5 * is_pos.float(), dim=1)

        # Target 1 indicates dist_an should be greater than dist_ap by margin
        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class HybridCrossModalLoss(nn.Module):
    """
    Hybrid Cross-Modality Loss combining Label-Smoothed Cross-Entropy Loss
    and Batch-Hard Triplet Loss for RGB and Synthetic-IR branches.
    """
    def __init__(
        self,
        num_classes: int,
        lambda_ce: float = 1.0,
        lambda_triplet: float = 1.0,
        margin: float = 0.3,
        epsilon: float = 0.1
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_triplet = lambda_triplet
        self.ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
        self.triplet_loss = BatchHardTripletLoss(margin=margin)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        rgb_logits: torch.Tensor,
        ir_feat: torch.Tensor,
        ir_logits: torch.Tensor,
        pids: torch.Tensor
    ) -> tuple:
        """
        Computes combined hybrid cross-modality loss over both RGB and IR samples.
        """
        # Concatenate RGB and IR embeddings and targets for cross-modality batch-hard mining
        combined_feat = torch.cat([rgb_feat, ir_feat], dim=0)
        combined_pids = torch.cat([pids, pids], dim=0)
        
        # Triplet Loss over combined cross-modal embeddings
        loss_triplet = self.triplet_loss(combined_feat, combined_pids)
        
        # Classification Cross-Entropy Loss on both RGB and IR logits
        loss_ce_rgb = self.ce_loss(rgb_logits, pids) if rgb_logits is not None else 0.0
        loss_ce_ir = self.ce_loss(ir_logits, pids) if ir_logits is not None else 0.0
        loss_ce = (loss_ce_rgb + loss_ce_ir) / 2.0
        
        total_loss = self.lambda_ce * loss_ce + self.lambda_triplet * loss_triplet
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_ce": loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce,
            "loss_triplet": loss_triplet.item()
        }


if __name__ == "__main__":
    loss_fn = HybridCrossModalLoss(num_classes=10)
    dummy_feat = torch.randn(16, 512)
    dummy_logits = torch.randn(16, 10)
    pids = torch.randint(0, 10, (16,))
    total_loss, metrics = loss_fn(dummy_feat, dummy_logits, dummy_feat, dummy_logits, pids)
    print(f"Total Loss: {total_loss.item():.4f}, Metrics: {metrics}")
