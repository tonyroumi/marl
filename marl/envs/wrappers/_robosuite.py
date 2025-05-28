from robosuite.wrappers import Wrapper
import gymnasium as gym
import torch

class RobosuiteWrapper(gym.Wrapper):
    """ Wrapper for robosuite environments to be used with gymnasium.
    """

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._get_observations(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._get_observations(obs)
        return obs, reward, terminated, truncated, info

    def _to_tensor(self, obs):
        return torch.from_numpy(obs).float()

    def __getattr__(self, name):
        if name == "spec":
            return self.env.spec
        return getattr(self.env, name)