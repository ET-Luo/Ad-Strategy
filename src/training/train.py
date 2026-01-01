import argparse
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.dataset_taobao import TaobaoSequenceDataset
from src.models.lstm_rec import LSTMRecModel
from src.models.transformer_rec import TransformerRecModel
from src.training.metrics import compute_auc


def get_model(
    model_type: Literal["lstm", "transformer"],
    num_items: int,
    max_seq_len: int,
) -> nn.Module:
    if model_type == "lstm":
        return LSTMRecModel(num_items=num_items)
    if model_type == "transformer":
        return TransformerRecModel(num_items=num_items, max_seq_len=max_seq_len)
    raise ValueError(f"Unknown model_type: {model_type}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                logits = model(batch)
                labels = batch["label"]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            labels = batch["label"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            labels = batch["label"]
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits)
            all_labels.append(labels)

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    auc = compute_auc(labels_cat, logits_cat)
    return avg_loss, auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM/Transformer on Taobao data")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/taobao_sequences.pt",
        help="Path to processed Taobao sequences file (torch.save)",
    )
    parser.add_argument("--model-type", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index to use, e.g. 0 or 3. Ignored if CUDA is not available.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    # Dataset internally splits into train/val by users
    train_dataset = TaobaoSequenceDataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        mode="train",
    )
    val_dataset = TaobaoSequenceDataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        mode="val",
    )

    pin_memory = device.type == "cuda"
    loader_common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_common_kwargs.update(
            {
                "prefetch_factor": 4,
                "persistent_workers": True,
            }
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )

    num_items = train_dataset.num_items
    model = get_model(args.model_type, num_items=num_items, max_seq_len=args.max_seq_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir=args.log_dir)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
        )

        global_step = epoch
        writer.add_scalar("loss/train", train_loss, global_step)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("metrics/auc", val_auc, global_step)

    writer.close()


if __name__ == "__main__":
    main()



