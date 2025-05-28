from dataclasses import dataclass
import torch
import os
from marl.storage.world_model_storage import WorldModelStorage

@dataclass
class Transition:
    """Container for a single environment transition.
    
    Stores all necessary information for policy gradient methods including
    observations, actions, rewards, and policy statistics.
    """
    actor_observations: torch.Tensor = None
    critic_observations: torch.Tensor = None
    actions: torch.Tensor = None
    rewards: torch.Tensor = None
    dones: torch.Tensor = None
    values: torch.Tensor = None
    actions_log_prob: torch.Tensor = None
    action_mean: torch.Tensor = None
    action_sigma: torch.Tensor = None

class RolloutStorage:
    """Storage buffer for collecting and processing rollout data.
    
    This class manages trajectories from multiple parallel environments, computes
    advantages using GAE (Generalized Advantage Estimation), and provides mini-batch
    sampling for policy optimization.

    Args:
      num_envs: Number of parallel environments
      num_transitions_per_env: Number of steps to collect per environment
      actor_obs_dim: Dimension of actor observations
      critic_obs_dim: Dimension of critic observations  
      action_dim: Dimension of action space
      device: PyTorch device for tensor storage
      world_model_buffer_size: Size of world model training buffer (optional)
      enable_world_model: Whether to collect data for world model training
    """
    
    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: torch.device,
        world_model_buffer_size: int = 1000000,
        enable_world_model: bool = False,
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.enable_world_model = enable_world_model

        # Initialize storage tensors with shape [num_steps, num_envs, feature_dim]
        self.observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.actor_obs_dim,
            device=self.device,
        )
        if self.critic_obs_dim is not None:
            self.critic_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                self.critic_obs_dim,
                device=self.device,
            )
        else:
            self.critic_observations = None

        self.rewards = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.action_dim,
            device=self.device,
        )
        
        self.dones = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        ).byte()

        # Value function and policy statistics
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, action_dim, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, action_dim, device=self.device)
        
        # Computed during advantage estimation
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.step = 0
        
        # World model components
        if self.enable_world_model:
            self.world_model_buffer = WorldModelStorage(
                capacity=world_model_buffer_size,
                state_dim=actor_obs_dim,
                action_dim=action_dim,
                device=device
            )
            
            # Store next observations for world model training
            self.next_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                self.actor_obs_dim,
                device=self.device,
            )

    def add(self, transition: Transition):
        """Add a new transition to the buffer.
        
        Args:
            transition: Transition object containing step data
            
        Raises:
            OverflowError: If buffer is full (call clear() first)
        """
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")
        
        # Copy transition data to buffer at current step
        self.observations[self.step].copy_(transition.actor_observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1     

    def add_next_observations(self, next_obs: torch.Tensor):
        """Add next observations for world model training.
        
        Args:
            next_obs: Next observations after taking actions
        """
        if self.enable_world_model and self.step > 0:
            self.next_observations[self.step - 1].copy_(next_obs)

    def clear(self):
        """Reset the buffer step counter to allow new data collection."""
        self.step = 0
        
    def transfer_to_world_model_buffer(self):
        """Transfer collected data to the world model buffer for long-term storage."""
        if not self.enable_world_model:
            return
            
        if self.step == 0:
            return
            
        # Flatten the rollout data
        states = self.observations[:self.step].flatten(0, 1)  # [steps*envs, obs_dim]
        actions = self.actions[:self.step].flatten(0, 1)      # [steps*envs, action_dim]
        next_states = self.next_observations[:self.step].flatten(0, 1)  # [steps*envs, obs_dim]
        rewards = self.rewards[:self.step].flatten(0, 1)      # [steps*envs, 1]
        dones = self.dones[:self.step].flatten(0, 1)          # [steps*envs, 1]
        
        # Add to world model buffer
        self.world_model_buffer.add_batch(states, actions, next_states, rewards, dones)
      
    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lambda_: float,
        normalize_advantage: bool = True,
    ):
        """Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_values: Value estimates for the final states
            gamma: Discount factor for future rewards
            lambda_: GAE parameter controlling bias-variance tradeoff
            normalize_advantage: Whether to normalize advantages to zero mean, unit variance
        """
        advantage = 0
        
        # Compute advantages backwards through time using GAE
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
                
            # Check if next state is terminal (done = 1 means terminal)
            next_is_not_terminal = 1.0 - self.dones[step].float()
            
            # Temporal difference error: δ = r + γV(s') - V(s)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            
            # GAE advantage: A = δ + γλA_{t+1}
            advantage = delta + next_is_not_terminal * gamma * lambda_ * advantage
            
            # Return: R = A + V (advantage + baseline)
            self.returns[step] = advantage + self.values[step]

        # Store advantages for training
        self.advantages = self.returns - self.values
        
        # Normalize advantages for training stability
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Generate mini-batches for policy optimization.
        
        Args:
            num_mini_batches: Number of mini-batches to create
            num_epochs: Number of epochs to iterate over the data
            
        Yields:
            Tuple of tensors for each mini-batch:
            (observations, critic_observations, actions, target_values, 
             advantages, returns, old_actions_log_prob, old_mu, old_sigma)
        """
        mini_batch_size = self.num_envs * self.num_transitions_per_env // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        
        # Flatten time and environment dimensions for batching
        observations = self.observations.flatten(0, 1)
        if self.critic_obs_dim is not None:
            critic_observations = self.critic_observations.flatten(0, 1)
        else:
            critic_observations = None

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        
        # Generate mini-batches for multiple epochs
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size
                batch_idx = indices[start_idx:end_idx]

                # Create mini-batch by indexing with random indices
                obs_batch = observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx] 
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield (obs_batch, critic_obs_batch, actions_batch, target_values_batch, 
                       advantages_batch, returns_batch, old_actions_log_prob_batch, 
                       old_mu_batch, old_sigma_batch)
    
    def world_model_batch_generator(self, batch_size: int):
        """Generate batches for world model training.
        
        Args:
            batch_size: Size of each training batch
            
        Yields:
            WorldModelTransition: Batch of world model training data
        """
        if not self.enable_world_model:
            raise ValueError("World model is not enabled. Set enable_world_model=True during initialization.")
            
        return self.world_model_buffer.sample(batch_size)
    
    def save_world_model_data(self, filepath: str):
        """Save world model buffer to disk.
        
        Args:
            filepath: Path to save the world model data
        """
        if not self.enable_world_model:
            raise ValueError("World model is not enabled.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.world_model_buffer.save(filepath)
    
    def load_world_model_data(self, filepath: str):
        """Load world model buffer from disk.
        
        Args:
            filepath: Path to load the world model data from
        """
        if not self.enable_world_model:
            raise ValueError("World model is not enabled.")
            
        self.world_model_buffer.load(filepath)
    
    def get_world_model_buffer_size(self) -> int:
        """Get current size of world model buffer."""
        if not self.enable_world_model:
            return 0
        return self.world_model_buffer.size
    
    def get_world_model_statistics(self) -> dict:
        """Get statistics about the world model buffer."""
        if not self.enable_world_model:
            return {}
            
        buffer = self.world_model_buffer
        return {
            'size': buffer.size,
            'capacity': buffer.capacity,
            'utilization': buffer.size / buffer.capacity,
            'mean_reward': buffer.rewards[:buffer.size].mean().item(),
            'std_reward': buffer.rewards[:buffer.size].std().item(),
            'done_rate': buffer.dones[:buffer.size].float().mean().item()
        }