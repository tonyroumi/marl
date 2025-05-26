from typing import Any, Dict
import gymnasium as gym

def is_gymnasium_env(env_id: str):
    """
    Check if the environment is a gymnasium environment
    """
    try:
        import gymnasium as gym
    except ImportError:
        return False
    from gymnasium.envs.registration import registry
  
    return env_id in registry

def env_factory(env_id: str, env_kwargs: Dict[str, Any]):
    """
    Create a gymnasium environment
    """
    return gym.make(
        env_id,
        **env_kwargs
    )