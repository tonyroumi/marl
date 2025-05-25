import os

from typing import Any, Dict, List

import torch
from torch.nn import Parameter

from marl.networks.network_factory import NetworkFactory

class Component:
    """
    A general-purpose component class that serves as a unified interface for flexible neural network
    operations. 
    
    Args:
        component_id: The ID of the component
        network_type: The type of network to use ("mlp", "cnn")
        network_class: The class of network to use ("actor", "critic", "actor_critic", "encoder")
        **network_kwargs: Additional arguments for network construction
    
    Raises:
        ValueError: If attempting to call act() on a critic-only network.
        ValueError: If attempting to call evaluate() on an actor-only network.
        ValueError: If attempting to call get_actions_log_prob() on a critic-only network.
        ValueError: If attempting to call get_action_mean() on a critic-only network.
        ValueError: If attempting to call get_action_std() on a critic-only network.
        ValueError: If attempting to call get_entropy() on a critic-only network.
        ValueError: If an invalid network_class or type is provided during initialization.
    """
    
    def __init__(
        self, 
        component_id: str,
        network_type: str,
        network_class: str,
        **network_kwargs: Any
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.component_id = component_id
        self.network_type = network_type
        self.network_class = network_class
        self.network_kwargs = network_kwargs
        self.is_frozen = False

        self._build_network()
    
    def _build_network(self):
        """ Build the network for the policy based on the network_class and network_type """
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

    def freeze(self):
        """
        Freeze all parameters in this component's network.
        Frozen parameters will not be updated during training.
        """
        for param in self.network.parameters():
            param.requires_grad = False
        self.is_frozen = True
        print(f"Component {self.component_id} ({self.network_class}) has been frozen")
    
    def unfreeze(self):
        """
        Unfreeze all parameters in this component's network.
        Parameters will be updated during training.
        """
        for param in self.network.parameters():
            param.requires_grad = True
        self.is_frozen = False
        print(f"Component {self.component_id} ({self.network_class}) has been unfrozen")
    
    def freeze_layers(self, layer_names: List[str]):
        """
        Freeze specific layers/modules by name.
        
        Args:
            layer_names: List of layer names to freeze (e.g., ['layer1', 'layer2.weight'])
        """
        frozen_count = 0
        for name, param in self.network.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_count += 1
                print(f"Frozen parameter: {name}")
        
        print(f"Frozen {frozen_count} parameters in component {self.component_id}")
    
    def get_trainable_parameters(self) -> List[Parameter]:
        """
        Get only the trainable (non-frozen) parameters.
        
        Returns:
            List of trainable parameters
        """
        return [param for param in self.network.parameters() if param.requires_grad]
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get information about the parameters in this component.
        
        Returns:
            Dictionary with parameter statistics
        """
        total_params = 0
        trainable_params = 0
        
        for param in self.network.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        return {
            'component_id': self.component_id,
            'network_class': self.network_class,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'is_frozen': self.is_frozen,
            'frozen_percentage': (total_params - trainable_params) / total_params * 100 if total_params > 0 else 0
        }

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network, producing raw network outputs.
        
        Args:
            input: Input to the network
            
        Returns:
            Tensor of raw network outputs
        """
    
        return self.network.forward(input)
    
    def act(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Get actions from the policy, either sampled (for exploration) or deterministic (for evaluation).
        
        Args:
            obs: Agent observations
            kwargs: Additional arguments for the network (deterministic)
            
        Returns:
            actions: The selected actions 
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            actions = self.network.act(obs, **kwargs)
        
        else:  # critic only
            raise ValueError(f"Cannot get actions from {self.network_class} network")

        return actions

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
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of the actions.

        Args:
            actions: The actions to get the log probability of
            
        Returns:
            log_prob: The log probability of the actions
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            return self.network.get_actions_log_prob(actions)
        else:
            raise ValueError("Cannot get actions log probability from critic-only network")
    
    def get_action_mean(self) -> torch.Tensor:
        """
        Get the mean of the action distribution.

        Returns:
            mean: The mean of the action distribution
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            return self.network.action_mean
        else:
            raise ValueError("Cannot get action mean from critic-only network")
    
    def get_action_std(self) -> torch.Tensor:
        """
        Get the standard deviation of the action distribution.

        Returns:
            std: The standard deviation of the action distribution
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            return self.network.action_std
        else:
            raise ValueError("Cannot get action std from critic-only network")
    
    def get_entropy(self) -> torch.Tensor:
        """
        Get the entropy of the action distribution.

        Returns:
            entropy: The entropy of the action distribution
        """
        if self.network_class == "actor" or self.network_class == "actor_critic":
            return self.network.entropy
        else:
            raise ValueError("Cannot get entropy from critic-only network")
    
    def parameters(self) -> Dict[str, List[Parameter]]:
        """ Get policy parameters """
        return self.network.parameters()
    
    def state_dict(self):
        """ Get the policy state dict for saving."""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load policy state dict."""
        self.network.load_state_dict(state_dict)
        
    def save(self, path):
        """Save policy to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        """Load policy from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint)