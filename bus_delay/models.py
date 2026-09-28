"""PyTorch model architectures used by the bus-delay experiments."""

import torch
from torch import nn

import process


class LSTM(nn.Module):
    """Legacy LSTM kept for loading version-one checkpoints."""

    def __init__(self, input_size: int = 6, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class EmbeddedDelayModel(nn.Module):
    """LSTM or MLP with categorical embeddings and a persistence residual.

    Route, direction, and next-stop IDs are embedded instead of treated as
    ordered numeric values. The zero-initialized output head starts at the
    strong persistence baseline: the latest observed delay.
    """

    def __init__(self, category_sizes, window=6, architecture="lstm",
                 hidden_size=64, num_layers=2, delay_mean=0.0, delay_std=1.0):
        super().__init__()
        if architecture not in ("lstm", "mlp"):
            raise ValueError("architecture must be lstm or mlp")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.architecture = architecture
        self.numeric_count = len(process.NUMERIC_COLUMNS)
        dimensions = [min(16, max(2, int(size ** 0.5))) for size in category_sizes]
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, dimension, padding_idx=0)
            for size, dimension in zip(category_sizes, dimensions)
        )
        input_width = self.numeric_count + sum(dimensions)
        if architecture == "lstm":
            self.encoder = nn.LSTM(
                input_width,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(window * input_width, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
            )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("delay_mean", torch.tensor(float(delay_mean)))
        self.register_buffer("delay_std", torch.tensor(float(delay_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(x[..., self.numeric_count + index].long())
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([x[..., :self.numeric_count], *embedded], dim=-1)
        if self.architecture == "lstm":
            encoded, _ = self.encoder(features)
            encoded = encoded[:, -1]
        else:
            encoded = self.encoder(features)
        delay_index = process.NUMERIC_COLUMNS.index("delay")
        latest_delay = x[:, -1, delay_index] * self.delay_std + self.delay_mean
        return latest_delay.unsqueeze(1) + self.head(encoded)
