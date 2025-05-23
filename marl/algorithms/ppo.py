import torch
import torch.nn as nn
import torch.optim as optim
from marl.storage.rollout_storage import Transition, RolloutStorage
from marl.algorithms.base import BaseAlgorithm
from typing import Dict, Any

class PPO(BaseAlgorithm):
  """
  Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).
  
  Adapted from: https://github.com/leggedrobotics/rsl_rl for multi-agent reinforcement learning.
  """
  def __init__(
      self,
      policy,
      agent_hyperparams: Dict[str, Dict[str, Any]],
      normalize_advantage_per_mini_batch: bool = False,
    ):
    self.policy = policy
    self.optimizers = {}
    self.storage = {}
    self.transitions = {agent_id: Transition() for agent_id in agent_hyperparams.keys()}

    self.agents = [agent_id for agent_id in agent_hyperparams.keys()]

    self.clip_param = {agent_id: params["clip_param"] for agent_id, params in agent_hyperparams.items()}
    self.num_learning_epochs = {agent_id: params["num_learning_epochs"] for agent_id, params in agent_hyperparams.items()}
    self.num_mini_batches = {agent_id: params["num_mini_batches"] for agent_id, params in agent_hyperparams.items()}
    self.value_loss_coef = {agent_id: params["value_loss_coef"] for agent_id, params in agent_hyperparams.items()}
    self.entropy_coef = {agent_id: params["entropy_coef"] for agent_id, params in agent_hyperparams.items()}
    self.gamma = {agent_id: params["gamma"] for agent_id, params in agent_hyperparams.items()}
    self.lambda_ = {agent_id: params["lambda_"] for agent_id, params in agent_hyperparams.items()}
    self.max_grad_norm = {agent_id: params["max_grad_norm"] for agent_id, params in agent_hyperparams.items()}
    self.use_clipped_value_loss = {agent_id: params["use_clipped_value_loss"] for agent_id, params in agent_hyperparams.items()}
    self.desired_kl = {agent_id: params["desired_kl"] for agent_id, params in agent_hyperparams.items()}
    self.schedule = {agent_id: params["schedule"] for agent_id, params in agent_hyperparams.items()}
    self.learning_rate = {agent_id: params["learning_rate"] for agent_id, params in agent_hyperparams.items()}
    self.num_transitions_per_env = {agent_id: params["num_transitions_per_env"] for agent_id, params in agent_hyperparams.items()}
    self.num_steps_per_env = {agent_id: params["num_steps_per_env"] for agent_id, params in agent_hyperparams.items()}
    self.total_timesteps = {agent_id: params["total_timesteps"] for agent_id, params in agent_hyperparams.items()}

    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    self._setup_optimizers()

  def _setup_optimizers(self):
    for agent_id in self.agents:
      self.optimizers[agent_id] = optim.Adam(
        self.policy.parameters(agent_id=agent_id),
        lr=self.learning_rate[agent_id]
      )

  def _init_storage(self, num_envs: int, agent_id: str):
    if agent_id in self.agents:
      self.storage[agent_id] = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=self.num_transitions_per_env[agent_id],
        actor_obs_dim=self.policy.components[agent_id].network_kwargs["actor_obs_dim"],
        critic_obs_dim=self.policy.components[agent_id].network_kwargs["critic_obs_dim"],
        action_dim=self.policy.components[agent_id].network_kwargs["num_actions"],
        device=self.policy.device
      )
    else:
      raise ValueError(f"Agent {agent_id} not found in agents list")
    
  def act(self, actor_obs, critic_obs, agent_id):
    self.transitions[agent_id].actions = self.policy.act(actor_obs, agent_id=agent_id).detach()
    self.transitions[agent_id].values = self.policy.evaluate(critic_obs, agent_id=agent_id).detach()
    self.transitions[agent_id].actions_log_prob = self.policy.get_actions_log_prob(self.transitions[agent_id].actions, agent_id).detach()
    self.transitions[agent_id].action_mean = self.policy.get_action_mean(agent_id).detach() 
    self.transitions[agent_id].action_sigma = self.policy.get_action_std(agent_id).detach()
    self.transitions[agent_id].actor_observations = actor_obs
    self.transitions[agent_id].critic_observations = critic_obs
    return self.transitions[agent_id].actions

  def process_env_step(self, rewards, dones, agent_id) -> None:
    self.transitions[agent_id].rewards = rewards#.clone()
    self.transitions[agent_id].dones = dones#.clone() #for single environment this is fine

    self.storage[agent_id].add(self.transitions[agent_id])
    self.transitions[agent_id] = Transition()


  def compute_returns(self, last_critic_obs, agent_id) -> None:
    last_values = self.policy.evaluate(last_critic_obs.clone(), agent_id=agent_id).detach()
    self.storage[agent_id].compute_returns(
      last_values,
      self.gamma[agent_id],
      self.lambda_[agent_id],
      not self.normalize_advantage_per_mini_batch
    )

  def update(self, agent_id: str) -> None:
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0

    generator = self.storage[agent_id].mini_batch_generator(self.num_mini_batches[agent_id], self.num_learning_epochs[agent_id])

    for (
        actor_obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        ) in generator:
      original_batch_size = actor_obs_batch.shape[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
      self.policy.act(actor_obs_batch, agent_id=agent_id)
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch, agent_id=agent_id)

      value_batch = self.policy.evaluate(critic_obs_batch, agent_id=agent_id)

      mu_batch = self.policy.get_action_mean(agent_id)[:original_batch_size]
      sigma_batch = self.policy.get_action_std(agent_id)[:original_batch_size]
      entropy_batch = self.policy.get_entropy(agent_id)[:original_batch_size]

      if self.desired_kl[agent_id] is not None and self.schedule[agent_id] == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
              torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
              + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
              / (2.0 * torch.square(sigma_batch))
              - 0.5,
              axis=-1,
          )
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          for param_group in self.optimizers[agent_id].param_groups:
            param_group["lr"] = self.learning_rate[agent_id]

      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param[agent_id], 1.0 + self.clip_param[agent_id]
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
          value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
              -self.clip_param[agent_id], self.clip_param[agent_id]
          )
          value_losses = (value_batch - returns_batch).pow(2)
          value_losses_clipped = (value_clipped - returns_batch).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
          value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef[agent_id] * value_loss - self.entropy_coef[agent_id] * entropy_batch.mean()

      self.optimizers[agent_id].zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.policy.parameters(agent_id=agent_id), self.max_grad_norm[agent_id])
      self.optimizers[agent_id].step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
    
    num_updates = self.num_learning_epochs[agent_id] * self.num_mini_batches[agent_id]
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates

    self.storage[agent_id].clear()

    loss_dict = {
      "value_loss": mean_value_loss,
      "surrogate_loss": mean_surrogate_loss,
      "entropy": mean_entropy,
    }

    return loss_dict
      


    

  
    
    