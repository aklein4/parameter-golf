from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from tqdm import tqdm

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import datasets


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_train_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    data = []
    for file in tqdm(files, desc="Loading training shards"):
        data.append(load_data_shard(file))

    return torch.cat(data).contiguous()


def main():
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")

    length = 2048

    train_data = load_train_tokens(train_files).cpu().detach().numpy().astype(np.uint16)
    val_data = load_validation_tokens(val_files, seq_len=1024).cpu().detach().numpy().astype(np.uint16)
    boundary_id = 1

    train_boundary = train_data == boundary_id
    val_boundary = val_data == boundary_id

    train_sequences = []
    current_length = 0
    for i, token in tqdm(enumerate(train_boundary), desc="Processing training data", total=len(train_boundary)):

        if token:
            if current_length >= length:
                train_sequences.append(train_data[i - current_length : i][:length])
            current_length = 1
        else:
            current_length += 1

    if current_length >= length:
        train_sequences.append(train_data[len(train_data) - current_length :][:length])

    val_sequences = []
    current_length = 0
    for i, token in tqdm(enumerate(val_boundary), desc="Processing validation data", total=len(val_boundary)):

        if token:
            if current_length >= length:
                val_sequences.append(val_data[i - current_length : i][:length])
            current_length = 1
        else:
            current_length += 1

    if current_length >= length:
        val_sequences.append(val_data[len(val_data) - current_length :][:length])

    print(f"Train tokens: {len(train_sequences) * length:_}")
    print(f"Validation tokens: {len(val_sequences) * length:_}")

    train_ds = datasets.Dataset.from_dict({"input_ids": train_sequences}, split="train")
    val_ds = datasets.Dataset.from_dict({"input_ids": val_sequences}, split="validation")

    train_ds = train_ds.shuffle(seed=42)
    val_ds = val_ds.shuffle(seed=42)

    train_ds.save_to_disk("./data/datasets/fineweb_2048_train")
    val_ds.save_to_disk("./data/datasets/fineweb_2048_val")


if __name__ == "__main__":
    main()