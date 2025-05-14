from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter

class BaseNetwork(nn.Module, ABC):
    """Abstract base class for all network architectures."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raise NotImplementedError("Subclasses must implement this method.")

    def init_weights(self, init_method: str = "orthogonal", gain: float = 1.0):
        """Initialize network weights."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        """Get network parameters."""
        pass
                    
    def save(self, path: str) -> None:
        """Save network parameters to disk."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load network parameters from disk."""
        self.load_state_dict(torch.load(path))


class BaseActorNetwork(BaseNetwork):
    """Base class for actor networks."""
    
    @abstractmethod
    def act(
        self, 
        obs: torch.Tensor, 
        **kwargs: Any
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute actions from observations.
        
        Args:
            obs: Tensor of observations
            deterministic: If True, use deterministic action selection
            
        Returns:
            actions: Tensor of actions
            info: Additional information (e.g., log probabilities)
        """
        pass


class BaseCriticNetwork(BaseNetwork):
    """Base class for critic networks."""
    
    @abstractmethod
    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute state values.
        
        Args:
            obs: Tensor of observations
            
        Returns:
            Tuple containing:
                - values: Tensor of estimated values
        """
        pass


class BaseActorCriticNetwork(BaseNetwork):
    """Base class for combined actor-critic networks."""
    
    @abstractmethod
    def act(
        self, 
        obs: torch.Tensor,
        **kwargs: Any
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute actions from observations.
        
        Args:
            obs: Tensor of observations
            **kwargs: Any (e.g. deterministic)
            
        Returns:
            Tuple containing:
                - actions: Tensor of actions
                - info: Dictionary with additional information (e.g., log probabilities)
        """
        pass
    
    @abstractmethod
    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute values from observations (and actions for Q-networks).
        
        Args:
            obs: Tensor of observations
            
        Returns:
            Tuple containing:
                - values: Tensor of estimated values
        """
        pass

    @abstractmethod
    def act_and_evaluate(
        self,
        obs: torch.Tensor,
        **kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute actions and values from observations.
        
        Args:
            obs: Tensor of observations
            
        Returns:
            Tuple containing:
                - actions: Tensor of actions
                - values: Tensor of estimated values
                - info: Dictionary with additional information (e.g., log probabilities)
        """
        pass

class BaseEncoderNetwork(BaseNetwork):
    """Base class for encoder networks."""
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass