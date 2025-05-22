
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
  def __init__(self, policy, **kwargs: Any):
    self.policy = policy
    
  @abstractmethod
  def act(self, obs, privileged_obs):
    pass

  @abstractmethod
  def learn(self):
    pass