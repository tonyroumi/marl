import torch
from robosuite.wrappers.wrapper import Wrapper


class TorchObsWrapper(Wrapper):
    def reset(self):
        obs = super().reset()
        return self._to_tensor(obs)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self._to_tensor(obs), reward, done, info
    
    def _get_observations(self):
        obs = self.env._get_observations()
        return self._to_tensor(obs)
    
    def _to_tensor(self, obs):
        return {key: torch.from_numpy(obs[key]).float() for key in obs}