
from marl.storage.base_buffer import BaseBuffer
from dataclasses import dataclass
import torch

@dataclass
class Transition:
  observations: torch.Tensor = None
  critic_observations: torch.Tensor = None
  actions: torch.Tensor = None
  rewards: torch.Tensor = None
  dones: torch.Tensor = None
  values: torch.Tensor = None
  actions_log_prob: torch.Tensor = None
  action_mean: torch.Tensor = None
  action_sigma: torch.Tensor = None
  
class PPOBuffer(BaseBuffer):
  def __init__(
      self,
      num_envs: int,
      num_transitions_per_env: int,
      actor_obs_dim: int,
      critic_obs_dim: int,
      action_dim: int,
      device: torch.device,
  ):
      # super().__init__(size, batch_size, device)
      self.device = device
      self.num_envs = num_envs
      self.num_transitions_per_env = num_transitions_per_env
      self.actor_obs_dim = actor_obs_dim
      self.critic_obs_dim = critic_obs_dim
      self.action_dim = action_dim

      self.observations = torch.zeros(
        num_transitions_per_env,
        num_envs,
        self.actor_obs_dim,
        device=self.device,
      )

  
      self.critic_observations = torch.zeros(
        num_transitions_per_env,
        num_envs,
        self.critic_obs_dim,
        device=self.device,
      )

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
      )

      self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
      self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
      self.mu = torch.zeros(num_transitions_per_env, num_envs, action_dim, device=self.device)
      self.sigma = torch.zeros(num_transitions_per_env, num_envs, action_dim, device=self.device)
      self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
      self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)


    
  def add(self, transition: Transition):
      if self.step >= self.num_transitions_per_env:
         raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")
      
      self.observations[self.step].copy_(transition.observations)
      if self.privileged_obs_dim is not None:
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
      self.actions[self.step].copy_(transition.actions)
      self.privileged_actions[self.step].copy_(transition.privileged_actions)
      self.rewards[self.step].copy_(transition.rewards)
      
      self.values[self.step].copy_(transition.values)
      self.dones[self.step].copy_(transition.dones)
      self.mu[self.step].copy_(transition.mu)
      self.sigma[self.step].copy_(transition.sigma)

      self.step += 1      
      
  def compute_returns(
      self,
      last_values: torch.Tensor,
      gamma: float,
      lambda_: float,
      normalize_advantage: bool = True,
  ):
     advantage = 0
     for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
          next_values = last_values
        else:
          next_values = self.values[step + 1]
          # 1 if we are not in a terminal state, 0 otherwise
          next_is_not_terminal = 1.0 - self.dones[step].float()
          # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
          delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
          # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
          advantage = delta + next_is_not_terminal * gamma * lambda_ * advantage
          # Return: R_t = A(s_t, a_t) + V(s_t)
          self.returns[step] = advantage + self.values[step]

     self.advantages = self.returns - self.values
     if normalize_advantage:
       self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
     mini_batch_size = self.num_envs * self.num_transitions_per_env // num_mini_batches
     indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
     observations = self.observations.flatten(0, 1)
     if self.privileged_obs_dim is not None:
        privileged_observations = self.privileged_observations.flatten(0, 1)
     else:
        privileged_observations = None


     actions = self.actions.flatten(0, 1)
     values = self.values.flatten(0, 1)
     returns = self.returns.flatten(0, 1)

     old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
     advantages = self.advantages.flatten(0, 1)
     old_mu = self.mu.flatten(0, 1)
     old_sigma = self.sigma.flatten(0, 1)
     
     for _ in range(num_epochs):
        for i in range(num_mini_batches):
          start_idx = i * mini_batch_size
          end_idx = (i+1) * mini_batch_size
          batch_idx = indices[start_idx:end_idx]

          obs_batch = observations[batch_idx]
          privileged_obs_batch = privileged_observations[batch_idx]
          actions_batch = actions[batch_idx]

          target_values_batch = values[batch_idx]
          returns_batch = returns[batch_idx]
          old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
          advantages_batch = advantages[batch_idx]
          old_mu_batch = old_mu[batch_idx]
          old_sigma_batch = old_sigma[batch_idx]

          yield obs_batch, privileged_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                )
          
    
        
        
        
      
      
      

      
      
      
      

      
      

    
    
    