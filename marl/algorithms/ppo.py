
import torch
import torch.nn as nn
import torch.optim as optim
from marl.storage.ppo_buffer import Transition, PPOBuffer

class PPO:
  def __init__(
      self
        # self,
        # policy,
        # num_learning_epochs=1,
        # num_mini_batches=1,
        # clip_param=0.2,
        # gamma=0.998,
        # lambda_=0.95,
        # value_loss_coef=1.0,
        # entropy_coef=0.0,
        # learning_rate=1e-3,
        # max_grad_norm=1.0,
        # use_clipped_value_loss=True,
        # schedule="fixed",
        # desired_kl=0.01,
        # device="cpu",
        # normalize_advantage_per_mini_batch=False,
    ):
    # self.policy = policy
    # self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    # self.storage = None
    # self.transition = Transition()

    # self.clip_param = clip_param
    # self.num_learning_epochs = num_learning_epochs
    # self.num_mini_batches = num_mini_batches
    # self.value_loss_coef = value_loss_coef
    # self.entropy_coef = entropy_coef
    # self.gamma = gamma
    # self.lambda_ = lambda_
    # self.max_grad_norm = max_grad_norm
    # self.use_clipped_value_loss = use_clipped_value_loss
    # self.desired_kl = desired_kl
    # self.schedule = schedule
    # self.learning_rate = learning_rate
    # self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
   pass
  def init_storage(
      self,
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      actions_shape,
      device
  ):
    return PPOBuffer(
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      actions_shape,
      device
    )

  def process_env_step(self, rewards, dones, infos):
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones.clone()

    self.storage.add(self.transition)
    self.policy.reset(dones)
    return Transition()

  def compute_returns(self, last_critic_obs):
    last_values_dict = self.policy.evaluate(last_critic_obs)

    self.storage.compute_returns(
      last_values_dict["values"],
      self.gamma,
      self.lambda_,
      self.normalize_advantage_per_mini_batch
    )
    

  
    
    