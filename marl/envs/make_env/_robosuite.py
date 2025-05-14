from typing import Any, Dict
import robosuite

def is_robosuite_env(env_id: str):
    """
    Check if the environment is a robosuite environment
    """
    try:
        import robosuite
    except ImportError:
        return False
    from robosuite.environments import ALL_ENVIRONMENTS

    return env_id in ALL_ENVIRONMENTS

def env_factory(env_id: str, env_kwargs: Dict[str, Any]):
    """
    Create a robosuite environment
    """
    return robosuite.make(
        env_id,
        **env_kwargs
    )