import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTCrossModalReID(nn.Module):
    """
    Vision Transformer (ViT) Backbone with Projection Head into Shared Embedding Space
    for Cross-Modality RGB-Infrared Person Re-Identification.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "vit_small_patch16_224",
        embed_dim: int = 512,
        drop_rate: float = 0.1,
        grad_checkpointing: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        
        # Load ViT backbone via timm (num_classes=0 returns extracted features)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            drop_rate=drop_rate
        )
        
        # Enable Gradient Checkpointing to conserve VRAM on 6GB RTX 4050
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
                print("[Model] Gradient Checkpointing enabled on ViT backbone.")
            except Exception as e:
                print(f"[Model] Could not enable gradient checkpointing: {e}")
                
        # Determine feature dimension from backbone
        in_features = self.backbone.num_features
        
        # Linear Projection Head + BatchNorm into shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Classifier Head for Cross-Entropy Training
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
            # Initialize weights
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized feature embeddings for cross-modality matching."""
        raw_feat = self.backbone(x)
        proj_feat = self.projection(raw_feat)
        norm_feat = F.normalize(proj_feat, p=2, dim=1)
        return norm_feat

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass during training.
        Returns:
            norm_feat: L2-normalized embedding tensor [B, embed_dim]
            logits: Classification logits [B, num_classes] (or None if no classifier)
        """
        raw_feat = self.backbone(x)
        proj_feat = self.projection(raw_feat)
        norm_feat = F.normalize(proj_feat, p=2, dim=1)
        
        if self.classifier is not None and self.training:
            logits = self.classifier(proj_feat)
            return norm_feat, logits
        return norm_feat, None


if __name__ == "__main__":
    model = ViTCrossModalReID(num_classes=751, backbone_name="vit_small_patch16_224", grad_checkpointing=True)
    dummy_input = torch.randn(4, 3, 224, 224)
    feat, logits = model(dummy_input)
    print(f"Feature shape: {feat.shape}, Logits shape: {logits.shape if logits is not None else None}")
