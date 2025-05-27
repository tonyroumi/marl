from typing import Callable, Optional
from functools import partial

import marl.envs.make_env._robosuite as _robosuite
import marl.envs.make_env._gymnasium as _gymnasium
from marl.envs.wrappers.common import EpisodeStatsWrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from robosuite.wrappers.gym_wrapper import GymWrapper

def wrap_env(env, wrappers=[]):
    """ Wrap the environment with the given wrappers. """
    for wrapper in wrappers:
        env = wrapper(env)
    return env

def make_env(
    env_id: str,
    env_type: str,
    num_envs: Optional[int] = 1,
    seed: Optional[int] = 0,
    record_video_path: Optional[str] = None,
    env_kwargs: dict = dict(),
    wrappers: list[Callable] = [],
):
    """ Make an environment. 
    
    Args:
        env_id: The id of the environment to make.
        env_type: The type of environment to make (only cpu atm).
        env_kwargs: The kwargs to pass to the environment.
        wrappers: The wrappers to wrap the environment with.
    """
    if env_type == "jax":
        raise NotImplementedError("Jax environment is not implemented yet")

    else:
        context = "fork"
        wrappers = [EpisodeStatsWrapper, *wrappers]


        if _robosuite.is_robosuite_env(env_id):
            env_factory = _robosuite.env_factory
            wrappers = [partial(GymWrapper, flatten_obs=False), *wrappers]
            context = "forkserver"
        elif _gymnasium.is_gymnasium_env(env_id):
            env_factory = _gymnasium.env_factory
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        if env_type == "gym:cpu":
            # create a vector env parallelized across CPUs
            vector_env_cls = partial(AsyncVectorEnv, context=context)
            if num_envs == 1:
                vector_env_cls = SyncVectorEnv
            env = vector_env_cls(
                [
                    env_factory(
                        env_id=env_id,
                        idx=idx,
                        env_kwargs=env_kwargs,
                        record_video_path=record_video_path,
                        wrappers=wrappers
                    )
                    for idx in range(num_envs)
                ]
            )
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        env.reset(seed=seed)

        return env
