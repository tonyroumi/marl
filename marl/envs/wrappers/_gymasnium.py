"""
This module contains wrappers specific to the Gymnasium library.
Its primary purpose is to standardize the observation space format returned by Gymnasium environments.

NOTE: 
In this codebase, all environments are expected to return observations as a dictionary,
with specific keys corresponding to observation components. However, some Gymnasium environments
return a single observation instead of a dictionary. This wrapper ensures compatibility by
wrapping the original observation in a dictionary under the key `'all_obs'`.
"""

from gymnasium import Wrapper
from gymnasium.spaces import Dict, Discrete


class GymnasiumWrapper(Wrapper):
    """
    A wrapper for Gymnasium environments that standardizes the observation format.
    """
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.action_space, Discrete):
            self.action_dim = env.action_space.n
        else:  # Continuous action space
            self.action_dim = env.action_space.shape[0]

        self.observation_space = Dict({
            'all_obs': env.observation_space
        })

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {'all_obs': obs}, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {'all_obs': obs}, info

    def close(self):
        return self.env.close()