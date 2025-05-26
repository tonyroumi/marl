from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Union, Iterator

from torch.nn import Parameter
import torch

class BasePolicy(ABC):
    """ 
    Abstract base class for all policies.

    This class provides the interface that all policies must implement.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """ 
        Perform a forward pass through the network.
        Produces raw network outputs (for modules that are not actor or critic).

        Args:
            obs: Agent observations, can be numpy array or PyTorch tensor

        Returns:
           Raw network outputs.
        """
        raise NotImplementedError("This method is not implemented for this policy.")
    
    @abstractmethod
    def act(
        self, 
        obs: torch.Tensor, 
        **kwargs: Any
        ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Get actions from the policy.
        
        Args:
            obs: Agent observations
            
        Returns:
            actions: The selected actions
        """
        pass
    
    def evaluate(
        self, 
        obs: torch.Tensor, 
        ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Get value estimates from the policy (for policies with critic components).
        
        Args:
            obs: Agent observations
            
        Returns:
            values: The estimated values

        Raises:
            NotImplementedError: If the policy does not implement a value function
        """
        raise NotImplementedError("This method is not implemented for this policy.")

    @abstractmethod
    def parameters(self) -> Union[Iterator[Parameter], Dict[str, List[Parameter]]]:
        """
        Get all parameters of the policy.

        Returns:
            parameters: The policy parameters
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