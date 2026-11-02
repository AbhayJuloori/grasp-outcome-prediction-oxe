"""Small 1D CNN over the early telemetry window (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TelemetryCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 1, width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=(0, 2), keepdims=True)
    sd = train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int = 2,
                epochs: int = 60, lr: float = 3e-3, seed: int = 0) -> np.ndarray:
    """Train on (N, C, T) arrays and return class probabilities for x_test.

    Binary tasks return P(class 1) with shape (N,); multiclass returns (N, n_classes).
    Statistics for standardisation come from the training fold only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_test = _standardize(x_train, x_test)
    binary = n_classes == 2
    model = TelemetryCNN(x_train.shape[1], 1 if binary else n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    xt = torch.from_numpy(x_train.astype(np.float32))
    if binary:
        yt = torch.from_numpy(y_train.astype(np.float32))
        pos = max(yt.mean().item(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    else:
        yt = torch.from_numpy(y_train.astype(np.int64))
        loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=gen).split(64):
            opt.zero_grad()
            out = model(xt[idx])
            loss = loss_fn(out.squeeze(1) if binary else out, yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x_test.astype(np.float32)))
        if binary:
            return torch.sigmoid(out.squeeze(1)).numpy()
        return torch.softmax(out, dim=1).numpy()
