import argparse
import time
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_taobao import TaobaoSequenceDataset


def parse_num_workers_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure DataLoader benchmark to tune num_workers."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/taobao_sequences.pt",
        help="Path to processed Taobao sequences file (torch.save).",
    )
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--num-workers-list",
        type=str,
        default="0,4,8,12,16,24,32",
        help="Comma-separated list of num_workers values to benchmark.",
    )
    args = parser.parse_args()

    num_workers_list = parse_num_workers_list(args.num_workers_list)

    print(
        f"Benchmarking DataLoader only: max_seq_len={args.max_seq_len}, "
        f"batch_size={args.batch_size}"
    )
    print(f"Data path: {args.data_path}")
    print(f"num_workers candidates: {num_workers_list}")
    print()

    dataset = TaobaoSequenceDataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        mode="train",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    for nw in num_workers_list:
        print(f"---- num_workers={nw} ----")
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": nw,
            "pin_memory": pin_memory,
        }
        if nw > 0:
            loader_kwargs.update(
                {
                    "prefetch_factor": 4,
                    "persistent_workers": True,
                }
            )

        loader = DataLoader(dataset, **loader_kwargs)

        start = time.perf_counter()
        for _ in tqdm(loader, desc=f"num_workers={nw}", leave=False):
            # We intentionally do nothing here to measure pure
            # data loading + CPU/memory/disk throughput.
            pass
        elapsed = time.perf_counter() - start
        print(f"num_workers={nw} -> elapsed={elapsed:.2f}s")
        print()


if __name__ == "__main__":
    main()


