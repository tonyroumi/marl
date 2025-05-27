from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from marl.storage.rollout_storage import Transition, RolloutStorage
from marl.algorithms.base import BaseAlgorithm

class PPO(BaseAlgorithm):
  """
  Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).
  
  Implementation adapted from https://github.com/leggedrobotics/rsl_rl for multi-agent 
  reinforcement learning scenarios.
  
  This class supports both single-agent and multi-agent training, with per-agent
  hyperparameters and storage management.
    
  Args:
      policy: Multi-agent policy network
      agent_hyperparams: Dictionary mapping agent_id to hyperparameter dictionaries
      normalize_advantage_per_mini_batch: Whether to normalize advantages per mini-batch
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

    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    self._setup_hyperparams(agent_hyperparams)
    self._setup_optimizers()

  def _setup_hyperparams(self, agent_hyperparams: Dict[str, Dict[str, Any]]) -> None:
    """ Setup hyperparams for each agent """
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
    self.total_timesteps = {agent_id: params["total_timesteps"] for agent_id, params in agent_hyperparams.items()}


  def _setup_optimizers(self):
    """ Setup optimizers for each agent """
    for agent_id in self.agents:
      self.optimizers[agent_id] = optim.Adam(
        self.policy.parameters(agent_id=agent_id),
        lr=self.learning_rate[agent_id]
      )

  def _init_storage(self, num_envs: int, num_transitions_per_env: int, agent_id: Optional[str] = None):
    """
    Initialize storage for agents.
    
    Args:
        num_envs: Number of environments
        agent_id: Specific agent to initialize storage for, or None to initialize for all agents
    """
    if agent_id is None:
        # Initialize storage for all agents
        for agent in self.agents:
            self._init_storage_single_agent(num_envs, num_transitions_per_env, agent)
    else:
        # Initialize storage for single agent
        self._init_storage_single_agent(num_envs, num_transitions_per_env, agent_id)

  def _init_storage_single_agent(self, num_envs: int, num_transitions_per_env: int, agent_id: str):
      """Initialize storage for a single agent"""
      if agent_id in self.agents:
          self.storage[agent_id] = RolloutStorage(
              num_envs=num_envs,
              num_transitions_per_env=num_transitions_per_env,
              actor_obs_dim=self.policy.components[agent_id].network_kwargs["actor_obs_dim"],
              critic_obs_dim=self.policy.components[agent_id].network_kwargs["critic_obs_dim"],
              action_dim=self.policy.components[agent_id].network_kwargs["num_actions"],
              device=self.policy.device
          )
      else:
          raise ValueError(f"Agent {agent_id} not found in agents list")
    
  def act(self, actor_obs, critic_obs=None, agent_id: Optional[str] = None):
    """
    Execute actions for agents.
    
    Args:
        actor_obs: Observations for actor (can be dict mapping agent_id->obs or single obs)
        critic_obs: Observations for critic (can be dict mapping agent_id->obs or single obs)
        agent_id: Specific agent to act for, or None to act for all agents
    
    Returns:
        If agent_id specified: actions tensor for that agent
        If agent_id is None: dict mapping agent_id -> actions tensor
    """
    if agent_id is None:
        # Act for all agents
        all_actions = {}
        
        # Handle case where observations are provided as dictionaries
        if isinstance(actor_obs, dict):
            for agent in self.agents:
                if agent in actor_obs:
                    agent_critic_obs = critic_obs[agent] if isinstance(critic_obs, dict) and agent in critic_obs else critic_obs
                    all_actions[agent] = self._act_single_agent(actor_obs[agent], agent_critic_obs, agent)
        else:
            # Same observations for all agents
            for agent in self.agents:
                all_actions[agent] = self._act_single_agent(actor_obs, critic_obs, agent)
        
        return all_actions
    else:
        # Act for single agent
        return self._act_single_agent(actor_obs, critic_obs, agent_id)

  def _act_single_agent(self, actor_obs, critic_obs, agent_id: str):
      """Execute action for a single agent"""
      # Use actor_obs as critic_obs if critic_obs not provided
      if critic_obs is None:
          critic_obs = actor_obs
          
      self.transitions[agent_id].actions = self.policy.act(actor_obs, agent_id=agent_id).detach()
      self.transitions[agent_id].values = self.policy.evaluate(critic_obs, agent_id=agent_id).detach()
      self.transitions[agent_id].actions_log_prob = self.policy.get_actions_log_prob(self.transitions[agent_id].actions, agent_id).detach()
      self.transitions[agent_id].action_mean = self.policy.get_action_mean(agent_id).detach() 
      self.transitions[agent_id].action_sigma = self.policy.get_action_std(agent_id).detach()
      self.transitions[agent_id].actor_observations = actor_obs
      self.transitions[agent_id].critic_observations = critic_obs
      return self.transitions[agent_id].actions

  def process_env_step(self, rewards, dones, agent_id: Optional[str] = None) -> None:
    """
    Process environment step for agents.
    
    Args:
        rewards: Rewards (can be dict mapping agent_id->reward or single reward)
        dones: Done flags (can be dict mapping agent_id->done or single done)
        agent_id: Specific agent to process for, or None to process for all agents
    """
    if agent_id is None:
        # Process for all agents
        if isinstance(rewards, dict) and isinstance(dones, dict):
            # Different rewards/dones for each agent
            for agent in self.agents:
                if agent in rewards and agent in dones:
                    self._process_env_step_single_agent(rewards[agent], dones[agent], agent)
        else:
            # Same rewards/dones for all agents
            for agent in self.agents:
                self._process_env_step_single_agent(rewards, dones, agent)
    else:
        # Process for single agent
        self._process_env_step_single_agent(rewards, dones, agent_id)

  def _process_env_step_single_agent(self, rewards, dones, agent_id: str) -> None:
      """Process environment step for a single agent"""
      self.transitions[agent_id].rewards = torch.from_numpy(rewards).clone()
      self.transitions[agent_id].dones = torch.from_numpy(dones).clone()

      self.storage[agent_id].add(self.transitions[agent_id])
      self.transitions[agent_id] = Transition()

  def compute_returns(self, last_critic_obs, agent_id: Optional[str] = None) -> None:
      """
      Compute returns for agents.
      
      Args:
          last_critic_obs: Last critic observations (can be dict mapping agent_id->obs or single obs)
          agent_id: Specific agent to compute returns for, or None to compute for all agents
      """
      if agent_id is None:
          # Compute returns for all agents
          if isinstance(last_critic_obs, dict):
              # Different observations for each agent
              for agent in self.agents:
                  if agent in last_critic_obs:
                      self._compute_returns_single_agent(last_critic_obs[agent], agent)
          else:
              # Same observations for all agents
              for agent in self.agents:
                  self._compute_returns_single_agent(last_critic_obs, agent)
      else:
          # Compute returns for single agent
          self._compute_returns_single_agent(last_critic_obs, agent_id)

  def _compute_returns_single_agent(self, last_critic_obs, agent_id: str) -> None:
      """Compute returns for a single agent"""
      last_values = self.policy.evaluate(last_critic_obs, agent_id=agent_id).detach()
      self.storage[agent_id].compute_returns(
          last_values,
          self.gamma[agent_id],
          self.lambda_[agent_id],
          not self.normalize_advantage_per_mini_batch
      )

  def update(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
    # If no agent_id specified, update all agents
    if agent_id is None:
        all_loss_dicts = {}
        for agent_id in self.agents:
            loss_dict = self._update_single_agent(agent_id)
            all_loss_dicts[agent_id] = loss_dict
        return all_loss_dicts
    else:
        # Update single agent
        return self._update_single_agent(agent_id)

  def _update_single_agent(self, agent_id: str) -> Dict[str, float]:
      """Update a single agent's parameters"""
      mean_value_loss = 0
      mean_surrogate_loss = 0
      mean_entropy = 0

      generator = self.storage[agent_id].mini_batch_generator(
          self.num_mini_batches[agent_id], 
          self.num_learning_epochs[agent_id]
      )

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

            if kl_mean > self.desired_kl[agent_id] * 2.0:
                self.learning_rate[agent_id] = max(1e-5, self.learning_rate[agent_id] / 1.5)
            elif kl_mean < self.desired_kl[agent_id] / 2.0 and kl_mean > 0.0:
                self.learning_rate[agent_id] = min(1e-2, self.learning_rate[agent_id] * 1.5)

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