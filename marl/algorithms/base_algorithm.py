from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from marl.policies import BasePolicy

class BaseAlgorithm(ABC):
    """Base class for all RL algorithms."""
    
    def __init__(self, policy: BasePolicy):
        self.policy = policy
        
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Any:
        """Update the algorithm with a batch of data.
        
        Args:
            batch: Dictionary containing trajectories and relevant information
            
        """
        pass
    
    @abstractmethod
    def act(self, observation: torch.Tensor, **kwargs) -> Any:
        """Get action from the algorithm's policy.
        
        Args:
            observation: Current observation
            **kwargs: Additional arguments (critic_obs)
            
        """
        pass
    