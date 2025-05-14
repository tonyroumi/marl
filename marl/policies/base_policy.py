from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List

import torch

class BasePolicy(ABC):
    """ 
    Abstract base class for all policies.
    This class provides the interface that all policies must implement.
    """

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """ 
        Perform a forward pass through the network.
        Produces raw network outputs (for modules that are not actor or critic).

        Args:
            obs: Agent observations, can be numpy array or PyTorch tensor

        Returns:
           Raw network outputs.
        """
        pass
    
    @abstractmethod
    def act(
        self, 
        obs: torch.Tensor, 
        **kwargs: Any
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get actions from the policy, either sampled (for exploration) or deterministic (for evaluation).
        
        Args:
            obs: Agent observations
            
        Returns:
            Tuple containing:
                - actions: The selected actions 
                - info: Dictionary with additional information such as log probabilities,
                  distribution parameters, and entropy
        """
        pass
    
    def evaluate(
        self, 
        obs: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Get value estimates from the policy (for policies with critic components).
        
        This method is not abstract as not all policies have value functions.
        Default implementation raises NotImplementedError.
        
        Args:
            obs: Agent observations
            
        Returns:
            Tuple containing:
                - values: The estimated values
                - info: Dictionary with additional value-related information
                
        Raises:
            NotImplementedError: If the policy does not implement a value function
        """
        raise NotImplementedError("This method is not implemented for this policy.")

    @abstractmethod
    def parameters(self) -> Dict[str, List[torch.Tensor]]:
        """
        Get all parameters of the policy.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save policy to disk.
        
        Args:
            path: Path to save the policy to
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load policy from disk.
        
        Args:
            path: Path to load the policy from
        """
        pass