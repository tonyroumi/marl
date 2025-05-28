"""
MARL Component Builders

This module provides functions to build MARL components (environments, policies, 
algorithms, and agents) from configuration files.
"""

from typing import Dict, Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from marl.agents.base_marl import BaseMARLAgent
from marl.agents.basic_marl_agent import BasicMARLAgent
from marl.envs.make_env.make_env import make_env
from marl.policies import MultiAgentPolicyBuilder
from marl.utils.utils import resolve_controller
from marl.algorithms.ppo import PPO

# =============================================================================
# Environment Builder
# =============================================================================

def build_env_from_config(config: DictConfig):
    """
    Build environment from configuration.
    
    Args:
        config: Environment configuration containing:
            - id: Environment identifier
            - type: Environment type
            - env_kwargs: Environment-specific arguments including controller_configs
    
    Returns:
        Constructed environment instance
    """
    env_id = config.get("id")
    env_type = config.get("type")
    num_envs = config.get("num_envs")
    seed = config.get("seed")
    max_episode_steps = config.get("max_episode_steps", 200)
    record_video_path = config.get("record_video_path")

    env_kwargs = config.get("env_kwargs", {})
    
    # Resolve controller configuration if present
    if "controller_configs" in env_kwargs:
        controller_config = resolve_controller(env_kwargs["controller_configs"])
        env_kwargs["controller_configs"] = controller_config

    return make_env(
        env_id=env_id,
        env_type=env_type,
        num_envs=num_envs,
        seed=seed,
        max_episode_steps=max_episode_steps,
        record_video_path=record_video_path,
        env_kwargs=env_kwargs
    )


# =============================================================================
# Policy Builder
# =============================================================================

def build_policy_from_config(config: DictConfig):
    """
    Build multi-agent policy from configuration.
    
    Args:
        config: Policy configuration containing:
            - components: Dictionary of component configurations
            - connections: Optional list of component connections
    
    Returns:
        Constructed multi-agent policy instance
    """
    builder = MultiAgentPolicyBuilder()
        
    # Add components from config
    for component_id, component_config in config["components"].items():
        builder.add_component(
            component_id=component_id,
            **component_config
        )
    
    # Add connections from config if specified
    if "connections" in config:
        for connection in config["connections"]:
            builder.add_connection(**connection)
    
    return builder.build()


# =============================================================================
# Algorithm Builder
# =============================================================================

def build_algorithm_from_config(policy, config: Dict[str, Any]):
    """
    Build algorithm from configuration.
    
    Args:
        policy: Multi-agent policy instance
        config: Algorithm configuration parameters
    
    Returns:
        Constructed algorithm instance (currently PPO)
    """
    return PPO(policy, config)


def parse_agent_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate individual agent configurations by merging global and agent-specific parameters.
    
    Args:
        config: Algorithm configuration containing:
            - global: Global parameters applied to all agents
            - agent_specific_hyperparams: Agent-specific parameter overrides
    
    Returns:
        Dictionary mapping agent names to their merged configurations
    """
    global_params = config['global']
    agent_specific = config['agent_specific_hyperparams']
    
    agent_configs = {}
    
    for agent_name, agent_params in agent_specific.items():
        # Start with global parameters
        agent_config = global_params.copy()
        
        # Override with agent-specific parameters
        agent_config.update(agent_params)
        
        agent_configs[agent_name] = agent_config
    
    return agent_configs


# =============================================================================
# Observation Configuration Helpers
# =============================================================================

def _extract_observation_keys(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract actor and critic observation keys from agent configuration.
    
    Args:
        config: Agent configuration containing observation specifications
    
    Returns:
        Tuple of (actor_obs_keys, critic_obs_keys) dictionaries
    """
    actor_obs_keys = {
        agent: config["agent"][agent]["actor_observations"] 
        for agent in config["agent"] if agent != "kwargs" and agent != "agent_class"
    }
    
    critic_obs_keys = {
        agent: config["agent"][agent].get("critic_observations", config["agent"][agent]["actor_observations"])
        for agent in config["agent"] if agent != "kwargs" and agent != "agent_class"
    }
    
    return actor_obs_keys, critic_obs_keys


def _extract_agent_kwargs(config: DictConfig) -> bool:
    """
    Extract agent keyword arguments from configuration.
    
    Args:
        config: Configuration containing agent kwargs
    
    Returns:
        Tuple of (normalize_observations, preprocess_observations) flags
    """
    agent_kwargs = config["agent"]["kwargs"]
    normalize_observations = agent_kwargs["normalize_observations"]
    
    return normalize_observations


# =============================================================================
# Main Builder Function
# =============================================================================

def build_from_config(config: DictConfig) -> Tuple[Any, BaseMARLAgent]:
    """
    Build all MARL components from configuration.
    
    This is the main factory function that constructs the complete MARL pipeline
    including environment, policy, algorithm, and agent from a unified configuration.
    
    Args:
        config: Complete MARL configuration containing:
            - environment: Environment configuration
            - policy: Policy configuration  
            - algorithm: Algorithm configuration
            - agent: Agent configuration including observation specs
            - device: Optional device specification
    
    Returns:
        Tuple of (environment, agent) ready for training
    """
    config = OmegaConf.to_container(config, resolve=True)
    # Build environment with wrapper
    env = build_env_from_config(config["environment"])
    policy = build_policy_from_config(config["policy"])
    num_transitions_per_env = config["algorithm"]["global"]["num_transitions_per_env"]
    
    # Build algorithm with parsed agent configurations
    agent_configs = parse_agent_configs(config["algorithm"])
    algorithm = build_algorithm_from_config(policy, agent_configs)
    
    # Extract observation configurations
    actor_obs_keys, critic_obs_keys = _extract_observation_keys(config)
    
    # Extract agent behavioral flags
    normalize_observations = _extract_agent_kwargs(config)
    
    # Create unified observation configuration
    observation_config = {
        "actor_obs_keys": actor_obs_keys,
        "critic_obs_keys": critic_obs_keys
    }
    agent_class = config["agent"]["agent_class"]
    
    # Determine device
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    
    # Build agent with all components
    if agent_class == "BasicMARLAgent":
        agent = BasicMARLAgent(
            env=env,
            policy=policy, 
            algorithm=algorithm,
            observation_config=observation_config,
            num_transitions_per_env=num_transitions_per_env,
            normalize_observations=normalize_observations,
            device=device
        )
    
    return env, agent