from collections import deque, namedtuple
import random as r
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

Transition = namedtuple("Transition", ["obs_dict", "action"])

class VAEBuf:
    """Replay buffer specifically designed for VAE training"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, obs_dict: Dict[str, np.ndarray], action: np.ndarray):
        """
        Store a transition in the buffer.
        
        Args:
            obs_dict: Dictionary of observations
            action: Action tensor/array of length 14 (7+7 for two robots)
        """
        
        # ensure action is a flat 1‑D array of shape (D,)
        a = np.asarray(action)
        a = a.reshape(-1)                # now (D,)
        self.buffer.append(Transition(obs_dict, a))

    def sample_batch(self, batch_size: int) -> Tuple[List[Dict], torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (list of obs_dicts, batched actions tensor)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
            
        batch = r.sample(self.buffer, batch_size)
        obs_dicts, actions = zip(*batch)
        
        # Stack into a NumPy array of shape (B, D) even if some were (1,D)
        arr = np.stack(actions, axis=0)            # shape maybe (B,1,D) or (B,D)        
        arr = arr.reshape(arr.shape[0], -1)        # force (B, D)
        actions = torch.as_tensor(arr, dtype=torch.float32)
        
        return list(obs_dicts), actions

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= min_size


    