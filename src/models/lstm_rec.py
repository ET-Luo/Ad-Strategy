from typing import Dict

import torch
import torch.nn as nn


class LSTMRecModel(nn.Module):
    """
    Simple LSTM-based sequential recommendation model.

    Inputs (from TaobaoSequenceDataset):
      - history_seq: [batch_size, seq_len] LongTensor of item ids
      - target_item: [batch_size] LongTensor of item ids

    Output:
      - logits: [batch_size] float tensor
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.target_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history_seq = batch["history_seq"]  # [B, L]
        target_item = batch["target_item"]  # [B]

        h_emb = self.item_embedding(history_seq)  # [B, L, E]
        _, (h_n, _) = self.lstm(h_emb)  # h_n: [num_layers, B, H]
        user_repr = h_n[-1]  # [B, H]

        t_emb = self.target_embedding(target_item)  # [B, E]
        x = torch.cat([user_repr, t_emb], dim=-1)  # [B, H+E]
        logits = self.mlp(x).squeeze(-1)  # [B]
        return logits


__all__ = ["LSTMRecModel"]



