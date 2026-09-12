import torch
import torch.nn.functional as F

def hybrid_loss(outputs, labels, lambda_source=1.0, lambda_reg=1e-4):
    fused_loss = F.cross_entropy(outputs["fused_logits"], labels)
    source_logits = outputs["source_logits"]
    masks = outputs["source_masks"]
    source_losses = []
    for s in range(source_logits.shape[1]):
        valid = masks[:, s]
        if valid.any():
            source_losses.append(F.cross_entropy(source_logits[valid, s], labels[valid]))
    source_loss = torch.stack(source_losses).mean() if source_losses else fused_loss.new_tensor(0.0)
    reg_loss = outputs["attention_weights"].pow(2).mean()
    total = fused_loss + lambda_source * source_loss + lambda_reg * reg_loss
    return total, {
        "fused_loss": float(fused_loss.detach().cpu()),
        "source_loss": float(source_loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
