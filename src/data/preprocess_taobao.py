"""
Preprocessing script for Taobao User Behavior data.

Goal:
  - Read raw behavior logs from `data/raw/UserBehavior.csv` (or a given path)
  - Sort behaviors by (user_id, timestamp)
  - Build per-user item interaction sequences
  - Build an item-id mapping to dense ids [1..num_items]
  - Save processed artifacts into `data/processed/taobao_sequences.pt`

Notes:
  - This script is intentionally simple and focuses on correctness and readability.
  - For the full Taobao dataset (~100M rows), you may want to:
      * Use pandas chunked reading
      * Restrict to a subset of users via `--max-users`
      * Optionally filter by behavior_type (e.g. clicks only)
"""

import argparse
import os
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch


def build_sequences(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: str,
    behavior_col: str | None = None,
    keep_behaviors: Tuple[str, ...] | None = None,
    max_users: int | None = None,
    min_seq_len: int = 2,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    if behavior_col is not None and keep_behaviors is not None:
        df = df[df[behavior_col].isin(keep_behaviors)]

    # Sort by user and timestamp
    df = df.sort_values([user_col, time_col])

    # Optionally limit number of users (for quicker experiments)
    if max_users is not None:
        unique_users = df[user_col].unique()[:max_users]
        df = df[df[user_col].isin(unique_users)]

    # Build item id mapping
    raw_item_ids = df[item_col].unique()
    item2id: Dict[int, int] = {int(item): idx + 1 for idx, item in enumerate(raw_item_ids)}

    # Map items to dense ids
    df["_item_id_dense"] = df[item_col].map(lambda x: item2id[int(x)])

    # Group by user to build sequences
    user2seq: Dict[int, List[int]] = {}
    for user_id, group in df.groupby(user_col):
        seq = group["_item_id_dense"].tolist()
        if len(seq) >= min_seq_len:
            user2seq[int(user_id)] = seq

    return user2seq, item2id


def save_processed(
    out_path: str,
    user2seq: Dict[int, List[int]],
    item2id: Dict[int, int],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obj: Dict[str, Any] = {
        "user2seq": user2seq,
        "item2id": item2id,
        "num_items": len(item2id),
    }
    torch.save(obj, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Taobao User Behavior data")
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/UserBehavior.csv",
        help="Path to raw Taobao behavior CSV",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/processed/taobao_sequences.pt",
        help="Path to save processed sequences (torch.save)",
    )
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="item_id")
    parser.add_argument(
        "--time-col",
        type=str,
        default="time_stamp",
        help="Timestamp column name (e.g. time_stamp or timestamp)",
    )
    parser.add_argument(
        "--behavior-col",
        type=str,
        default="behavior_type",
        help="Behavior type column name (if exists)",
    )
    parser.add_argument(
        "--keep-behaviors",
        type=str,
        default="pv,cart,fav,buy",
        help="Comma-separated behavior types to keep; ignored if column missing",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=100000,
        help="Max number of users to keep (for faster experiments)",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=2,
        help="Minimum sequence length to keep a user",
    )
    args = parser.parse_args()

    if not os.path.exists(args.raw_path):
        raise FileNotFoundError(f"Raw data not found at {args.raw_path}")

    print(f"Loading raw data from {args.raw_path} ...")
    # The official Taobao User Behavior data from Tianchi has NO header row
    # and columns in the following order:
    #   user_id, item_id, cat_id, behavior_type, time_stamp
    # We explicitly assign column names here to avoid header-related issues.
    col_names = [
        args.user_col,
        args.item_col,
        "cat_id",
        args.behavior_col,
        args.time_col,
    ]
    df = pd.read_csv(args.raw_path, names=col_names, header=None)

    behavior_col = args.behavior_col if args.behavior_col in df.columns else None
    keep_behaviors = tuple(b.strip() for b in args.keep_behaviors.split(",")) if behavior_col else None

    print("Building user sequences and item mapping ...")
    user2seq, item2id = build_sequences(
        df=df,
        user_col=args.user_col,
        item_col=args.item_col,
        time_col=args.time_col,
        behavior_col=behavior_col,
        keep_behaviors=keep_behaviors,
        max_users=args.max_users,
        min_seq_len=args.min_seq_len,
    )

    print(f"Users kept: {len(user2seq)}, items: {len(item2id)}")
    print(f"Saving processed data to {args.out_path} ...")
    save_processed(args.out_path, user2seq=user2seq, item2id=item2id)
    print("Done.")


if __name__ == "__main__":
    main()


