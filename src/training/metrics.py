from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_auc(labels: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Compute AUC from labels and logits (before sigmoid).
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        # This happens if all labels are the same in a batch/dataset
        return float("nan")


__all__ = ["compute_auc"]



