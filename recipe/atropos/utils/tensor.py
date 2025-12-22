# Copyright 2025 Nous Research
# Licensed under the Apache License, Version 2.0
"""Tensor utilities."""

import torch
from typing import Union


def pad_sequences(
    sequences: list[list[Union[int, float]]],
    max_len: int,
    pad_value: Union[int, float] = 0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32 if isinstance(pad_value, float) else torch.long

    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=dtype)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=dtype)

    return padded
