
from abc import ABC, abstractmethod

class BaseBuffer(ABC):
  def __init__(self, size: int, batch_size: int, device: str):
    self.size = size
    self.batch_size = batch_size
    self.device = device
    self.step = 0

  @abstractmethod
  def add(self, transition: dict):
    pass

  def clear(self):
    self.step = 0

  
    