from typing import List, Dict, Any, Tuple, Iterator

import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import Parameter

from marl.networks.base_networks import BaseActorNetwork, BaseCriticNetwork, BaseActorCriticNetwork, BaseNetwork
from marl.utils.utils import resolve_nn_activation

class MLPActorNetwork(BaseActorNetwork):
    """MLP Actor Network."""
    
    def __init__(
        self, 
        actor_obs_dim: int, 
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        init_noise_std = 1.0,
        noise_std_type="scalar"
        ):
        """
        Initalize MLP actor network.

        Args:
            observation_dim: Dimension of observation space
            num_actions: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function to use
            init_noise_std: Initial standard deviation of noise
            noise_std_type: Type of noise standard deviation
        """
        super().__init__()
        
        self.actor_obs_dim = actor_obs_dim
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims

        self.activation = resolve_nn_activation(activation)

        self.init_noise_std = init_noise_std
        self.noise_std_type = noise_std_type

        self._init_net()
    
    def _init_net(self):
        """Initialize the network layers."""
        actor_layers = []
        actor_dims = [self.actor_obs_dim] + self.actor_hidden_dims
        for i in range(len(actor_dims) - 1):
            actor_layers.append(nn.Linear(actor_dims[i], actor_dims[i+1]))
            actor_layers.append(self.activation)
        actor_layers.append(nn.Linear(actor_dims[-1], self.num_actions))
        
        self.actor = nn.Sequential(*actor_layers)
        
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(torch.ones(self.num_actions) * self.init_noise_std)
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(self.init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        #Action Distribution (populated in update_distribution)
        self.distribution = None
        #disable args validation for speedup
        Normal.set_default_validate_args(False)
    
    def update_distribution(self, obs: torch.Tensor):
        """Update the action distribution."""
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std.expand_as(mean))
        self.distribution = Normal(mean, std)
    
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from the network.
        
        Args:
            obs: Agent observations
            deterministic: If True, return mean action instead of sampling

        Returns:
            Tuple containing:
                - actions: The selected actions
                - info: Dictionary with additional information (log_prob, mean, std, entropy)
        """
        info = {}
        self.update_distribution(obs)
        if deterministic:
            action = self.distribution.mean
        else:
            action = self.distribution.sample()
        info["log_prob"] = self.get_actions_log_prob(action)
        info["action_mean"] = self.action_mean
        info["action_std"] = self.action_std
        info["entropy"] = self.entropy

        return action, info
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities of actions.
        
        Args:
            actions: Actions to get log probabilities of

        Returns:
            Log probabilities of actions
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get actions from the network.
        
        Args:
            obs: Agent observations

        Returns:
            Actions from the network
        """
        return self.actor(obs)
    
    @property
    def action_mean(self) -> torch.Tensor: 
        """Mean of the action distribution."""
        return self.distribution.mean
    
    @property
    def action_std(self) -> torch.Tensor:
        """Standard deviation of the action distribution."""
        return self.distribution.stddev
    
    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def parameters(self) -> Iterator[Parameter]:
        """Parameters of the network."""
        return self.actor.parameters()
                    
            
class MLPCriticNetwork(BaseCriticNetwork):
    """MLP-based Critic Network."""
    
    def __init__(
        self, 
        critic_obs_dim: int, 
        critic_out_dim: int = 1,
        critic_hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        ):
        """
        Initialize MLP critic network.

        Args:
            observation_dim: Dimension of observation space
            critic_output_dim: Dimension of critic output
            hidden_dims: Dimensions of hidden layers
            activation: Activation function to use
        """
        super().__init__()

        self.critic_obs_dim = critic_obs_dim
        self.critic_out_dim = critic_out_dim
        self.critic_hidden_dims = critic_hidden_dims
        self.activation = resolve_nn_activation(activation)

        self._init_net()
    
    def _init_net(self):
        """Initialize the network layers."""

        critic_layers = []
        critic_dims = [self.critic_obs_dim] + self.critic_hidden_dims
        for i in range(len(critic_dims) - 1):
            critic_layers.append(nn.Linear(critic_dims[i], critic_dims[i+1]))
            critic_layers.append(self.activation)
        
        critic_layers.append(nn.Linear(critic_dims[-1], self.critic_out_dim))

        self.critic = nn.Sequential(*critic_layers)

    def evaluate(
        self,
        obs: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the critic network.
        
        Args:
            obs: Agent observations
            actions: Actions (unused for V-function critic)
            
        Returns:
            Dictionary containing value estimates
        """
        return self.critic(obs)
    
    def parameters(self) -> Iterator[Parameter]:
        return self.critic.parameters()
    
class MLPActorCriticNetwork(BaseActorCriticNetwork):
    """MLP-based Actor-Critic Network."""
    
    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        num_actions: int,
        critic_out_dim: int = 1,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        ):
        """
        Initialize MLP actor-critic network.

        Args:
            actor_obs_dim: Dimension of observation space for actor network
            critic_obs_dim: Dimension of observation space for critic network
            num_actions: Dimension of action space
            critic_output_dim: Dimension of critic output
            actor_hidden_dims: Dimensions of hidden layers
            critic_hidden_dims: Dimensions of hidden layers
            activation: Activation function to use
        """
        super().__init__()

        self.actor = MLPActorNetwork(
            actor_obs_dim=actor_obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = MLPCriticNetwork(
            critic_obs_dim=critic_obs_dim,
            critic_out_dim=critic_out_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation
        )

    
    def act_and_evaluate(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the actor-critic network.
        
        Args:
            obs: Agent observations
            
        Returns:
            Tuple containing:
                - Dictionary containing both actor outputs (distribution parameters) 
                and critic outputs (value estimates)
                - Dictionary with additional information (log_prob, mean, std, entropy)
        """
        # Actor outputs
        actor_outputs, actor_info = self.actor.act(obs, deterministic)
        
        # Critic output
        critic_outputs = self.critic.evaluate(obs)
        
        result = {
            "action": actor_outputs,
            "value": critic_outputs
        }
        
        return result, actor_info

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get actions from the actor component.
        
        Args:
            obs: Agent observations
            deterministic: If True, return mean action instead of sampling
            
        Returns:
            Tuple containing:
                - actions: The selected actions
                - info: Dictionary with additional information (log_prob, mean, std, entropy)
        """
        return self.actor.act(obs, deterministic)
    
    def evaluate(
        self,
        obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Get values from the critic component.
        
        Args:
            obs: Agent observations
            
        Returns:
            Tuple containing:
                - values: The estimated values
        """
        return self.critic.evaluate(obs)
    
    def parameters(self) -> Dict[str, List[Parameter]]:
        """Parameters of the network."""
        return {"actor": list(self.actor.parameters()), "critic": list(self.critic.parameters())}
        
class MLPEncoderNetwork(BaseNetwork):
    """MLP-based Encoder Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """
        Initialize MLP encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dims: Dimensions of hidden layers
            activation: Activation function to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = resolve_nn_activation(activation)

        self._init_net()

    def _init_net(self):
        """Initialize the network layers."""
        encoder_layers = []
        dims = [self.input_dim] + self.hidden_dims

        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(self.activation)

        encoder_layers.append(nn.Linear(dims[-1], self.output_dim))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder network."""
        return self.encoder(obs)

    def init_weights(self, init_method: str = "orthogonal", gain: float = 1.0):
        """Initialize the weights of the network.
        
        Raises:
            ValueError: If init_method is not "orthogonal", "xavier_uniform", or "kaiming_uniform"
        """
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                if init_method == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain)
                elif init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Unknown init_method: {init_method}")
                nn.init.zeros_(m.bias)

    def parameters(self) -> Iterator[Parameter]:
        """Parameters of the network."""
        return self.encoder.parameters()
        