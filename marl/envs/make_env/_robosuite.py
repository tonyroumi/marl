from typing import Any, Dict, List, Callable

import robosuite
from gymnasium.wrappers import RecordVideo
import gym

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

def env_factory(env_id: str, idx: int, record_video_path: str = None, env_kwargs: Dict[str, Any] = {}, wrappers: List[Callable] = []):
    """
    Create a robosuite environment wrapped for gymnasium
    """
    def _init():
        env = robosuite.make(
            env_id,
            **env_kwargs
        )
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
        return env
    return _init

  