from typing import Callable

import marl.envs.make_env._robosuite as _robosuite
import marl.envs.make_env._gymnasium as _gymnasium

def wrap_env(env, wrappers=[]):
    """ Wrap the environment with the given wrappers. """
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def make_env(
    env_id: str,
    env_type: str,
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

    if _robosuite.is_robosuite_env(env_id):
        env = _robosuite.env_factory(env_id, env_kwargs)
    elif _gymnasium.is_gymnasium_env(env_id):
        env = _gymnasium.env_factory(env_id, env_kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    env = wrap_env(env, wrappers)

    return env
