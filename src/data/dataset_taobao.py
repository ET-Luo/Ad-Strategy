import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _pad_or_truncate(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    # left-pad with 0 (padding idx)
    return [0] * (max_len - len(seq)) + seq


class TaobaoSequenceDataset(Dataset):
    """
    Dataset for Taobao User Behavior sequences based on preprocessed data.

    Preprocessed file (torch.save) should contain:
      - user2seq: Dict[user_id, List[item_id_dense]]
      - item2id: Dict[raw_item_id, dense_id]
      - num_items: int

    Each sample is:
      - history_seq: LongTensor [max_seq_len]
      - target_item: LongTensor [] (item id)
      - label:       FloatTensor [] (1.0 positive, 0.0 negative)

    For every positive (history -> next item), we generate one negative
    sample by sampling a random item that does not equal the positive item.
    """

    def __init__(
        self,
        data_path: str = "data/processed/taobao_sequences.pt",
        max_seq_len: int = 50,
        mode: str = "train",
        neg_ratio: int = 1,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Processed Taobao data not found at {data_path}. "
                f"Please run src/data/preprocess_taobao.py first."
            )

        obj = torch.load(data_path)
        user2seq: Dict[int, List[int]] = obj["user2seq"]
        self.num_items: int = int(obj["num_items"])
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.neg_ratio = max(1, int(neg_ratio))

        # Offline expansion: build dense arrays of histories and positive targets
        # so that __getitem__ becomes a simple index lookup + (cheap) neg sampling.
        histories: List[List[int]] = []
        targets: List[int] = []
        for _, seq in user2seq.items():
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                history = _pad_or_truncate(seq[:pos], self.max_seq_len)
                histories.append(history)
                targets.append(seq[pos])

        self.histories = np.asarray(histories, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)

        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(self.targets))
        rng.shuffle(all_indices)

        split_idx = int(len(all_indices) * (1.0 - val_ratio))
        if self.mode == "train":
            self.indices = all_indices[:split_idx]
        elif self.mode == "val":
            self.indices = all_indices[split_idx:]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.rng = rng

    def __len__(self) -> int:
        # we generate neg_ratio negatives per positive on the fly
        return len(self.indices) * (1 + self.neg_ratio)

    def _sample_negative(self, exclude_item: int) -> int:
        # Sample until we get an item different from exclude_item
        while True:
            item = int(self.rng.integers(1, self.num_items + 1))
            if item != exclude_item:
                return item

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map global idx back to (base_pos_idx, is_negative_idx)
        base_idx = idx // (1 + self.neg_ratio)
        offset = idx % (1 + self.neg_ratio)

        sample_idx = int(self.indices[base_idx])
        history = self.histories[sample_idx]
        pos_item = int(self.targets[sample_idx])

        if offset == 0:
            target_item = pos_item
            label = 1.0
        else:
            target_item = self._sample_negative(exclude_item=pos_item)
            label = 0.0

        return {
            "history_seq": torch.as_tensor(history, dtype=torch.long),
            "target_item": torch.as_tensor(target_item, dtype=torch.long),
            "label": torch.as_tensor(label, dtype=torch.float32),
        }


__all__ = ["TaobaoSequenceDataset"]



