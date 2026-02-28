from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset

from transformer_mppi.controllers.transformer import TransformerModel
from transformer_mppi.training.artifacts import TransformerArtifacts


class SequenceDataset(Dataset):
    def __init__(self, src: np.ndarray, target: np.ndarray):
        self.src = src
        self.target = target

    def __len__(self) -> int:
        return self.src.shape[0]

    def __getitem__(self, idx: int):
        src = torch.tensor(self.src[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)

        tgt_input = torch.zeros_like(target)
        tgt_input[1:, :] = target[:-1, :]
        return src, tgt_input, target


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


def train_transformer_model(
    input_sequences: np.ndarray,
    target_sequences: np.ndarray,
    horizon: int,
    k_history: int,
    hidden_size: int,
    num_layers: int,
    nhead: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[TransformerArtifacts, TrainingHistory]:
    if input_sequences.ndim != 3:
        raise ValueError("input_sequences must have shape (N, k, input_size)")
    if target_sequences.ndim != 3:
        raise ValueError("target_sequences must have shape (N, horizon, output_size)")

    input_size = input_sequences.shape[-1]
    output_size = target_sequences.shape[-1]

    x_flat = input_sequences.reshape(-1, input_size)
    y_flat = target_sequences.reshape(-1, output_size)

    n_quantiles_x = min(1000, x_flat.shape[0])
    n_quantiles_y = min(1000, y_flat.shape[0])
    input_scaler = QuantileTransformer(n_quantiles=n_quantiles_x, output_distribution="uniform", random_state=seed)
    output_scaler = QuantileTransformer(n_quantiles=n_quantiles_y, output_distribution="uniform", random_state=seed)

    input_scaler.fit(x_flat)
    output_scaler.fit(y_flat)

    x_scaled = input_scaler.transform(x_flat).reshape(input_sequences.shape)
    y_scaled = output_scaler.transform(y_flat).reshape(target_sequences.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled,
        y_scaled,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nhead=nhead,
        dropout=dropout,
        device=device,
    ).to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_state = None
    epochs_without_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for src, tgt_input, target in train_loader:
            src = src.permute(1, 0, 2).to(device)
            tgt_input = tgt_input.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)

            optimizer.zero_grad()
            outputs = model(src, tgt_input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * src.shape[1]

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt_input, target in val_loader:
                src = src.permute(1, 0, 2).to(device)
                tgt_input = tgt_input.permute(1, 0, 2).to(device)
                target = target.permute(1, 0, 2).to(device)

                outputs = model(src, tgt_input)
                loss = criterion(outputs, target)
                val_loss += loss.item() * src.shape[1]

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts = TransformerArtifacts(
        model=model,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        horizon=horizon,
        k_history=k_history,
        input_size=input_size,
        output_size=output_size,
        model_config={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "nhead": nhead,
            "dropout": dropout,
        },
    )
    history = TrainingHistory(train_losses=train_losses, val_losses=val_losses)
    return artifacts, history
