from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

class BaseAlgorithm(ABC):
    """Base class for all RL algorithms."""

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Any:
        pass
    
    @abstractmethod
    def act(self, observation: torch.Tensor, **kwargs) -> Any:
        """Get action from the algorithm's policy.
        
        Args:
            observation: Current observation
            **kwargs: Additional arguments (critic_obs)
            
        """
        pass