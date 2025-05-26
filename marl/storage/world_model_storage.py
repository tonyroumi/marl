import torch
from dataclasses import dataclass

@dataclass
class WorldModelTransition:
    """Container for world model training data."""
    states: torch.Tensor = None
    actions: torch.Tensor = None
    next_states: torch.Tensor = None
    rewards: torch.Tensor = None
    dones: torch.Tensor = None
    
class WorldModelStorage:
    """Dedicated buffer for world model training data."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.next_states = torch.zeros(capacity, state_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)
        
        self.position = 0
        self.size = 0
        
    def add_batch(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor,
          dones: torch.Tensor):
        """Add a batch of transitions to the world model buffer."""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            self.states[self.position] = states[i]
            self.actions[self.position] = actions[i]
            self.next_states[self.position] = next_states[i]
            self.rewards[self.position] = rewards[i]
            self.dones[self.position] = dones[i]
            
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> WorldModelTransition:
        """Sample a random batch for world model training."""
        if self.size < batch_size:
            batch_size = self.size
            
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return WorldModelTransition(
            states=self.states[indices],
            actions=self.actions[indices],
            next_states=self.next_states[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices]
        )
    
    def save(self, filepath: str):
        """Save buffer to disk."""
        data = {
            'states': self.states[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'next_states': self.next_states[:self.size].cpu(),
            'rewards': self.rewards[:self.size].cpu(),
            'dones': self.dones[:self.size].cpu(),
            'position': self.position,
            'size': self.size
        }
        torch.save(data, filepath)
    
    def load(self, filepath: str):
        """Load buffer from disk."""
        data = torch.load(filepath, map_location=self.device)
        size = data['size']
        self.states[:size] = data['states'].to(self.device)
        self.actions[:size] = data['actions'].to(self.device)
        self.next_states[:size] = data['next_states'].to(self.device)
        self.rewards[:size] = data['rewards'].to(self.device)
        self.dones[:size] = data['dones'].to(self.device)
        self.position = data['position']
        self.size = size