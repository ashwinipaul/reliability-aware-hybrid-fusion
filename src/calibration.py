import torch
import torch.nn.functional as F
from torch import nn

class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-3)

@torch.no_grad()
def entropy_from_logits(logits, temperature=1.0, eps=1e-8):
    probs = F.softmax(logits / float(temperature), dim=-1)
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1), probs

def fit_temperature(logits, labels, init_temperature=1.0, max_iter=50):
    scaler = TemperatureScaler(init_temperature).to(logits.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scaler.temperature.clamp_(0.05, 20.0)
    return float(scaler.temperature.detach().cpu())
