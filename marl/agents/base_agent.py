
from abc import ABC, abstractmethod
from typing import Any
import torch

class BaseAgent(ABC):
  def __init__(self, policy):
    self.policy = policy
    
  @abstractmethod
  def act(self, obs: torch.Tensor, critic_obs: torch.Tensor):
    pass

  @abstractmethod
  def learn(self):
    pass