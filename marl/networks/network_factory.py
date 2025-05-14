from typing import Dict, Any, List

from marl.networks.base_networks import BaseActorNetwork, BaseCriticNetwork, BaseActorCriticNetwork, BaseEncoderNetwork
from marl.networks.mlp_networks import MLPActorNetwork, MLPCriticNetwork, MLPActorCriticNetwork, MLPEncoderNetwork

class NetworkFactory:
    """Factory class for creating different types of networks."""

    @staticmethod
    def create_actor_network(network_type: str, 
                            actor_obs_dim: int,
                            num_actions: int,
                            **network_kwargs: Dict[str, Any]
                            ) -> BaseActorNetwork:
        """
        Create an actor network based on the specified type.
        
        Args:
            network_type: Type of network to create ("mlp", "cnn")
            actor_obs_dim: Dimension of observation space for actor network
            num_actions: Dimension of action space
            : Additional arguments for network construction

        Raises:
            ValueError: If network_type is not "mlp"

        Returns:
            An instance of the actor network
        """
        if network_type.lower() == "mlp":
            return MLPActorNetwork(
                actor_obs_dim=actor_obs_dim,
                num_actions=num_actions,
                **network_kwargs
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        
    @staticmethod
    def create_critic_network(
        network_type: str,
        critic_obs_dim: int,
        critic_out_dim: int,
        **network_kwargs: Dict[str, Any]
        ) -> BaseCriticNetwork:
        """
        Create a critic network based on the specified type.
        
        Args:
            network_type: Type of network to create ("mlp", "cnn")
            critic_obs_dim: Dimension of observation space for critic network
            critic_out_dim: Dimension of critic output
            : Additional arguments for network construction
        
        Raises:
            ValueError: If network_type is not "mlp"
            
        Returns:
            An instance of the critic network
        """
        if network_type.lower() == "mlp":
            return MLPCriticNetwork(
                critic_obs_dim=critic_obs_dim,
                critic_out_dim=critic_out_dim, 
                **network_kwargs
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        
    @staticmethod
    def create_actor_critic_network(
        network_type: str,
        actor_obs_dim: int,
        critic_obs_dim: int,
        num_actions: int,
        critic_out_dim: int = 1,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        **network_kwargs: Dict[str, Any]
        ) -> BaseActorCriticNetwork:
        """
        Create an actor-critic network based on the specified type.
        
        Args:
            network_type: Type of network to create ("mlp", "cnn")
            actor_obs_dim: Dimension of observation space for actor network
            critic_obs_dim: Dimension of observation space for critic network
            num_actions: Dimension of action space
            critic_out_dim: Dimension of critic output
            : Additional arguments for network construction
        
        Raises:
            ValueError: If network_type is not "mlp"
            
        Returns:
            An instance of the actor-critic network
        """
        if network_type.lower() == "mlp":
            return MLPActorCriticNetwork(
                actor_obs_dim=actor_obs_dim,
                critic_obs_dim=critic_obs_dim,
                num_actions=num_actions,
                critic_out_dim=critic_out_dim,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                **network_kwargs
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        
    @staticmethod
    def create_encoder_network(
        network_type: str,
        input_dim: int,
        output_dim: int,
        **network_kwargs: Dict[str, Any]
        ) -> BaseEncoderNetwork:
        """
        Create an encoder network based on the specified type.

        Args:
            network_type: Type of network to create ("mlp", "cnn")
            input_dim: Dimension of input space
            output_dim: Dimension of output space
            : Additional arguments for network construction
            
        Raises:
            ValueError: If network_type is not "mlp"
            
        Returns:
            An instance of the encoder network
        """
        if network_type.lower() == "mlp":
            return MLPEncoderNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                **network_kwargs
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
