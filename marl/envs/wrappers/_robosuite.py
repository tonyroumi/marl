from robosuite.wrappers import Wrapper
import torch

class RobosuiteWrapper(Wrapper):
    """ Wrapper for robosuite environments to be used with gymnasium.
    
    By default robosuite returns more environments then the gymnasium wrapper for some reason. 
    """

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._get_observations(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_observations(obs)
        return obs, reward, terminated, truncated, info
    
    def _get_observations(self, obs):
        return obs

    def _to_tensor(self, obs):
        return torch.from_numpy(obs).float()