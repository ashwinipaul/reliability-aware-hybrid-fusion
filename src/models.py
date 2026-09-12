import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class SourceEncoder(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=True, dropout=0.0):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.project = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.project(self.backbone(x))

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, masks=None):
        scores = self.attention(features).squeeze(-1)
        if masks is not None:
            scores = scores.masked_fill(~masks, -1e9)
        weights = F.softmax(scores, dim=1)
        if masks is not None:
            weights = weights * masks.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return fused, weights, scores

class SharedClassifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class ReliabilityAwareDecisionFusion(nn.Module):
    def __init__(self, epsilon=1e-8, temperature=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.temperature = temperature

    def forward(self, source_logits, masks):
        probs = F.softmax(source_logits / self.temperature, dim=-1)
        p = probs.clamp_min(self.epsilon)
        entropy = -(p * p.log()).sum(dim=-1)
        reliability = 1.0 / (entropy + self.epsilon)
        reliability = reliability.masked_fill(~masks, -1e9)
        weights = F.softmax(reliability, dim=1)
        weights = weights * masks.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        fused_probs = (probs * weights.unsqueeze(-1)).sum(dim=1)
        return fused_probs, weights, entropy, probs

class ReliabilityAwareHybridFusion(nn.Module):
    def __init__(
        self,
        num_sources,
        num_classes=2,
        embedding_dim=1024,
        attention_hidden_dim=256,
        pretrained=True,
        dropout=0.0,
        temperature=1.0,
        epsilon=1e-8,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleList([
            SourceEncoder(embedding_dim, pretrained, dropout)
            for _ in range(num_sources)
        ])
        self.feature_fusion = AdaptiveFeatureFusion(embedding_dim, attention_hidden_dim)
        # The manuscript describes a shared classification head for source-wise
        # and fused predictions.
        self.classifier = SharedClassifier(embedding_dim, num_classes)
        self.decision_fusion = ReliabilityAwareDecisionFusion(epsilon, temperature)

    def forward(self, sources, source_masks=None):
        device = next(self.parameters()).device
        available = [x for x in sources if x is not None]
        if not available:
            raise ValueError("At least one source must be available.")
        batch_size = available[0].shape[0]
        if source_masks is None:
            source_masks = [
                torch.ones(batch_size, dtype=torch.bool, device=device)
                if x is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
                for x in sources
            ]

        features, masks, source_logits = [], [], []
        for s, x in enumerate(sources):
            if x is None:
                features.append(torch.zeros(batch_size, self.embedding_dim, device=device))
                masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
                source_logits.append(torch.zeros(batch_size, self.num_classes, device=device))
                continue
            x = x.to(device, non_blocking=True)
            f = self.encoders[s](x)
            logits = self.classifier(f)
            m = source_masks[s].to(device)
            f = torch.where(m.unsqueeze(-1), f, torch.zeros_like(f))
            logits = torch.where(m.unsqueeze(-1), logits, torch.zeros_like(logits))
            features.append(f)
            masks.append(m)
            source_logits.append(logits)

        feature_tensor = torch.stack(features, dim=1)
        mask_tensor = torch.stack(masks, dim=1)
        fused_features, attention_weights, _ = self.feature_fusion(feature_tensor, mask_tensor)
        fused_logits = self.classifier(fused_features)
        source_logit_tensor = torch.stack(source_logits, dim=1)
        final_probs, reliability_weights, entropy, calibrated_probs = self.decision_fusion(
            source_logit_tensor, mask_tensor
        )
        return {
            "features": feature_tensor,
            "fused_features": fused_features,
            "attention_weights": attention_weights,
            "source_logits": source_logit_tensor,
            "fused_logits": fused_logits,
            "source_probs": calibrated_probs,
            "entropy": entropy,
            "reliability_weights": reliability_weights,
            "final_probs": final_probs,
            "final_logits": torch.log(final_probs.clamp_min(1e-8)),
            "source_masks": mask_tensor,
        }
