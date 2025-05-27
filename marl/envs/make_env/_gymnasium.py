from typing import Any, Dict, List, Callable

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit

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

def env_factory(env_id: str, idx: int, record_video_path: str = None, env_kwargs: Dict[str, Any] = {}, wrappers: List[Callable] = []):
    """
    Create a gymnasium environment
    """
    def _init():
        env = gym.make(
            env_id,
            **env_kwargs
        )
        env = TimeLimit(env, max_episode_steps=200)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
        return env
    return _init