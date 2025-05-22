from marl.policies import MultiAgentPolicyBuilder
from marl.envs.make_env.make_env import make_env
from marl.utils.utils import resolve_controller
# from marl.agents.single_agent import SingleAgent

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

def build_agent_from_config(config):
    """Build agent from configuration."""
    policy = build_policy_from_config(config["policy"])
    if config["agent_type"] == "single_agent":
        return SingleAgent(policy, **config["agent_kwargs"])
    else:
        raise ValueError(f"Invalid agent type: {config['agent_type']}")