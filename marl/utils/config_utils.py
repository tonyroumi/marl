from marl.policies import MultiAgentPolicyBuilder
from marl.envs.make_env.make_env import make_env
from marl.utils.utils import resolve_controller
from marl.agents.standard_agent import StandardAgent
from marl.algorithms.ppo import PPO
from marl.envs.wrappers.common import TorchObsWrapper
from typing import Dict, Any

def build_env_from_config(config):
    """Build environment from configuration."""
    env_id = config.get("id")
    env_type = config.get("type")
    env_kwargs = config.get("env_kwargs")
    controller_config = resolve_controller(env_kwargs.get("controller_configs"))
    env_kwargs["controller_configs"] = controller_config

    return make_env(env_id, env_type, env_kwargs)

def build_policy_from_config(config):
    """Build components from configuration."""
    builder = MultiAgentPolicyBuilder()
        
    # Add components from config
    for component_id, component_config in config["components"].items():
        builder.add_component(
            component_id=component_id,
            **component_config
        )
    
    # Add connections from config
    if "connections" in config:
        for connection in config["connections"]:
            builder.add_connection(**connection)
    
    # Build the policy
    return builder.build()

def build_algorithm_from_config(policy, config):
    """Build algorithm from configuration."""
    return PPO(policy, config)

def build_agent_from_config(config):
    """Build agent from configuration."""
    env = build_env_from_config(config["environment"])
    env = TorchObsWrapper(env)
    policy = build_policy_from_config(config["policy"])
    agent_configs = parse_agent_configs(config["algorithm"])
    algorithm = build_algorithm_from_config(policy, agent_configs)
    actor_obs_keys = {agent: config["agent"][agent]["actor_observations"] for agent in config["agent"] if agent != "kwargs"}
    critic_obs_keys = {agent: config["agent"][agent]["critic_observations"] for agent in config["agent"] if agent != "kwargs"}
    normalize_observations = config["agent"]["kwargs"]["normalize_observations"]
    preprocess_observations = config["agent"]["kwargs"]["preprocess_observations"]
   
    return env, StandardAgent(env,policy, algorithm, actor_obs_keys, critic_obs_keys, normalize_observations, preprocess_observations)


def parse_agent_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Generate individual agent configs by merging global and agent-specific params."""
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