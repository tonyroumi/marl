from abc import ABC, abstractmethod
from typing import Dict, Any
from marl.agents.base_agent import BaseAgent

class BaseAlgorithm(ABC):
    """Base class for all RL algorithms."""
        
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Any:
        """Update the algorithm with a batch of data.
        
        Args:
            batch: Dictionary containing trajectories and relevant information
            
        """
        pass
    