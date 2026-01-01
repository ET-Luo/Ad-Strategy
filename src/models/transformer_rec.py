from typing import Dict

import torch
import torch.nn as nn


class TransformerRecModel(nn.Module):
    """
    Simplified Transformer-based sequential recommendation model
    (SASRec/BST-style).

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
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.target_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history_seq = batch["history_seq"]  # [B, L]
        target_item = batch["target_item"]  # [B]

        bsz, seq_len = history_seq.size()
        positions = torch.arange(seq_len, device=history_seq.device).unsqueeze(0)

        h_emb = self.item_embedding(history_seq) + self.pos_embedding(positions)
        # For now we do not use padding mask; later we can add it when real data is wired.
        enc_out = self.encoder(h_emb)  # [B, L, E]
        user_repr = enc_out[:, -1, :]  # [B, E] use last position

        t_emb = self.target_embedding(target_item)  # [B, E]
        x = torch.cat([user_repr, t_emb], dim=-1)  # [B, 2E]
        logits = self.mlp(x).squeeze(-1)  # [B]
        return logits


__all__ = ["TransformerRecModel"]





