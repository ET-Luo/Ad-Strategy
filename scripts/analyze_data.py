import os
import torch
import numpy as np
import collections
from typing import Dict, List

def analyze_sequence_lengths(data_path: str = "data/processed/taobao_sequences.pt"):
    # 尝试自动定位路径（如果相对路径失败，尝试绝对路径）
    if not os.path.exists(data_path):
        abs_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "taobao_sequences.pt")
        if os.path.exists(abs_path):
            data_path = abs_path
        else:
            # 如果还是找不到，尝试你之前的完整路径，但去掉重复的部分
            potential_path = "/home/qw765731/projects/RC-System/Ad-Strategy/data/processed/taobao_sequences.pt"
            if os.path.exists(potential_path):
                data_path = potential_path

    print(f"Loading data from {data_path}...")
    obj = torch.load(data_path)
    user2seq: Dict[int, List[int]] = obj["user2seq"]
    
    lengths = [len(seq) for seq in user2seq.values()]
    lengths = np.array(lengths)
    
    num_users = len(lengths)
    total_records = np.sum(lengths)
    
    print("\n--- Summary Statistics ---")
    print(f"Total Users: {num_users}")
    print(f"Total Behavior Records: {total_records}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print(f"Median Length: {np.median(lengths):.2f}")
    print(f"Min Length: {np.min(lengths)}")
    print(f"Max Length: {np.max(lengths)}")
    print(f"Std Dev: {np.std(lengths):.2f}")
    
    print("\n--- Percentiles ---")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(lengths, p)}")
        
    print("\n--- Length Distribution (Top 10) ---")
    counter = collections.Counter(lengths)
    total = len(lengths)
    for length, count in counter.most_common(10):
        print(f"Length {length:3d}: {count:6d} users ({count/total*100:5.2f}%)")

    # Grouped distribution
    print("\n--- Grouped Distribution ---")
    bins = [0, 5, 10, 20, 50, 100, 200, 500, float('inf')]
    labels = []
    for i in range(len(bins) - 1):
        if bins[i+1] == float('inf'):
            labels.append(f">{bins[i]}")
        else:
            labels.append(f"{bins[i]+1}-{int(bins[i+1])}")
            
    bin_counts = collections.defaultdict(int)
    for length in lengths:
        for i in range(len(bins) - 1):
            if bins[i] < length <= bins[i+1]:
                bin_counts[labels[i]] += 1
                break
    
    for label in labels:
        count = bin_counts[label]
        print(f"{label:8s}: {count:6d} users ({count/total*100:5.2f}%)")

if __name__ == "__main__":
    analyze_sequence_lengths()

