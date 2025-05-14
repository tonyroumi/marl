import os

from typing import Any, Dict, List

import torch
from torch.nn import Parameter

from marl.policies import BasePolicy
from marl.networks.network_factory import NetworkFactory

class Policy(BasePolicy):
    """
    A general-purpose policy that maps observations to actions.
    
    This can be used for a single agent or as a component in a multi-agent system.
    """
    
    def __init__(
        self, 
        component_id: str,
        network_type: str,
        network_class: str,
        **network_kwargs: Any
        ):
        """
        Initialize a policy.
        
        Args:
            component_id: The ID of the policy
            network_type: The type of network to use ("mlp", "cnn")
            network_class: The class of network to use ("actor", "critic", "actor_critic", "encoder")
            **network_kwargs: Additional arguments for network construction
        ):
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.component_id = component_id
        self.network_type = network_type
        self.network_class = network_class
        self.network_kwargs = network_kwargs

        self._build_network()
    
    def _build_network(self):
        """ Build the network for the policy """
        if self.network_class == "actor":
            self.network = NetworkFactory.create_actor_network(
                network_type=self.network_type,
                **self.network_kwargs
            )
        elif self.network_class == "critic":
            self.network = NetworkFactory.create_critic_network(
                network_type=self.network_type,
                **self.network_kwargs
            )
        elif self.network_class == "actor_critic":
            self.network = NetworkFactory.create_actor_critic_network(
                network_type=self.network_type,
                **self.network_kwargs
            )
        elif self.network_class == "encoder":
            self.network = NetworkFactory.create_encoder_network(
                network_type=self.network_type,
                **self.network_kwargs
            )
        else:
            raise ValueError(f"Invalid network class: {self.network_class}")
        
        self.network.to(self.device)


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network, producing raw network outputs.
        
        Args:
            obs: Agent observations
            
        Returns:
            Dictionary of raw network outputs from the underlying network
        """
    
        return self.network.forward(obs)
    
    def act(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Get actions from the policy, either sampled (for exploration) or deterministic (for evaluation).
        
        Args:
            obs: Agent observations
            
        Returns:
            Tuple containing:
                - actions: The selected actions 
                - info: Dictionary with additional information
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            actions, info = self.network.act(obs, **kwargs)
        
        else:  # critic only
            raise ValueError(f"Cannot get actions from {self.network_class} network")

        return actions, info

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get values based on observations.
        
        Args:
            obs: Observations
            
        Returns:
            values: Value estimates
        """
        # Forward through network
        if self.network_class == "critic" or self.network_class == "actor_critic":
            values = self.network.evaluate(obs)
        else:
            raise ValueError("Cannot get values from actor-only network")
        
        return values
    
    def parameters(self) -> Dict[str, List[Parameter]]:
        """ Get policy parameters """
        return self.network.parameters()
    
    def state_dict(self):
        """ Get the policy state dict for saving."""
        return {
            "network": self.network.state_dict(),
            "network_class": self.network_class
        }
    
    def load_state_dict(self, state_dict):
        """Load policy state dict."""
        self.network.load_state_dict(state_dict["network"])
        self.network_class = state_dict["network_class"]
        
    def save(self, path):
        """Save policy to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "network_class": self.network_class,
        }, path)
    
    def load(self, path):
        """Load policy from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.network_class = checkpoint["network_class"]
        