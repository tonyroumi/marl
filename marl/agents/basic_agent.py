import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from marl.policies.base_policy import BasePolicy
from marl.algorithms.base import BaseAlgorithm


class BasicAgent:
    """"
    Basic agent class that handles multi-agent reinforcement learning.
    
    Manages observation processing, normalization, and coordinates between policy,
    algorithm, and environment components.

    Args:
      env: Environment instance
      policy: Policy network instance
      algorithm: Learning algorithm instance
      observation_config: Configuration mapping agent/critic names to observation keys
      normalize_observations: Whether to apply observation normalization
      preprocess_observations: Whether to preprocess observations
      logger: Optional logger instance
      device: PyTorch device for computations

    """
    def __init__(
        self, 
        env: Any,
        policy: BasePolicy, 
        algorithm: BaseAlgorithm,
        observation_config: Dict[str, List[str]],
        num_transitions_per_env: int,
        normalize_observations: bool,
        preprocess_observations: bool,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None,
        ):
      self.env = env
      self.env.num_envs = 1 #Placeholder for now
      self.policy = policy
      self.algorithm = algorithm
      self.logger = logger

      self.observation_config = observation_config
      self.normalize_observations = normalize_observations
      self.preprocess_observations = preprocess_observations
      self.num_transitions_per_env = num_transitions_per_env
      self.device = device

      self.actor_obs_keys = self.observation_config['actor_obs_keys']
      self.critic_obs_keys = self.observation_config['critic_obs_keys'] \
        if self.observation_config['critic_obs_keys'] else self.actor_obs_keys

      self._post_init()

    def _post_init(self):
        """Initialize derived properties after main initialization."""
        obs = self.env.reset()
        
        # Compute observation dimensions
        self.num_actor_obs = {}
        for agent, keys in self.actor_obs_keys.items():
            self.num_actor_obs[agent] = sum(obs[key].shape[-1] for key in keys)
        
        self.num_critic_obs = {}
        for critic, keys in self.critic_obs_keys.items():
            self.num_critic_obs[critic] = sum(obs[key].shape[-1] for key in keys)
        
        self.agents = list(self.actor_obs_keys.keys())
        self.critics = list(self.critic_obs_keys.keys())
        
        # Initialize buffers and normalizers
        self.buffers = {}
        self._init_buffers()
        self._init_normalizers()

    def _init_normalizers(self):
        """Initialize observation normalizers based on config"""
        device = self.device
        
        if self.normalize_observations:
            from marl.networks import EmpiricalNormalization
            
            self.actor_obs_normalizer = {}
            self.critic_obs_normalizer = {}
            
            for agent in self.agents: #Right now there are separate normalizers for actor and critic. we may want this to be the same
                self.actor_obs_normalizer[agent] = EmpiricalNormalization(
                    shape=[self.num_actor_obs[agent]], 
                    until=1.0e8
                ).to(device)
            
            for critic in self.critics:
                self.critic_obs_normalizer[critic] = EmpiricalNormalization(
                    shape=[self.num_critic_obs[critic]], 
                    until=1.0e8
                ).to(device)
        else:
            self.actor_obs_normalizer = {agent: torch.nn.Identity() for agent in self.agents}
            self.critic_obs_normalizer = {critic: torch.nn.Identity() for critic in self.critics}

    def _init_buffers(self):
        self.buffers = self.algorithm._init_storage(self.env.num_envs, self.num_transitions_per_env)
    
    def process_observations(self, all_obs: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        """
        Process observations by selecting and concatenating desired observations.
        
        Args:
            all_obs: Dictionary containing all observations from the environment
            
        Returns:
            Tuple of (actor_observations, critic_observations) dictionaries for each agent and critic
        """
        actor_obs = {}
        critic_obs = {}
        
        # Process actor observations
        for agent in self.agents:
            agent_obs = torch.cat(
                [all_obs[key] for key in self.actor_obs_keys[agent]], 
                dim=-1
            ).to(self.device)
            actor_obs[agent] = agent_obs
        
        # Process critic observations
        for critic in self.critics:
            critic_obs_tensor = torch.cat(
                [all_obs[key] for key in self.critic_obs_keys[critic]], 
                dim=-1
            ).to(self.device)
            critic_obs[critic] = critic_obs_tensor
        
        return actor_obs, critic_obs
    
    def _normalize_observations(self, actor_obs: Dict, critic_obs: Dict) -> Tuple[Dict, Dict]:
        """Apply normalization to observations."""
        for agent in self.agents:
            actor_obs[agent] = self.actor_obs_normalizer[agent](actor_obs[agent])
        
        for critic in self.critics:
            critic_obs[critic] = self.critic_obs_normalizer[critic](critic_obs[critic])
        
        return actor_obs, critic_obs
    
    def _collect_rollouts(self, actor_obs: Dict, critic_obs: Dict, num_transitions: int = 24) -> Tuple[Dict, Dict]:
        """Collect rollout experiences from environment interactions."""
        actions = np.zeros((len(self.agents), self.env.action_dim // len(self.agents)))
        
        with torch.inference_mode():
            for _ in range(num_transitions):
                # Get actions from all agents
                for i, agent_id in enumerate(self.agents):
                    actions[i] = self.algorithm.act(
                        actor_obs=actor_obs[agent_id],
                        critic_obs=critic_obs[agent_id],
                        agent_id=agent_id
                    ).cpu().numpy()

                # Step environment
                obs, rewards, dones, infos = self.env.step(actions.squeeze())
                print(rewards)
                actor_obs, critic_obs = self.process_observations(obs)
                actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)

                # Process environment step for each agent
                for agent in self.agents:
                    self.algorithm.process_env_step(rewards, dones, agent)
        
        return actor_obs, critic_obs
      
    def learn(self, total_iterations: int = 1000) -> None:
        """
        Main training loop.
        
        Args:
            total_iterations: Number of training iterations to run
        """
        
        # Get initial observations
        all_obs = self.env.reset()
        actor_obs, critic_obs = self.process_observations(all_obs)
        self.train_mode() #Switch to training mode

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        curr_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        curr_ep_len = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        
        for iteration in range(total_iterations):
            start_time = time.time()
            with torch.inference_mode():
            
                # Collect rollouts
                actor_obs, critic_obs = self._collect_rollouts(actor_obs, critic_obs, self.num_transitions_per_env)
                collection_time = time.time() - start_time
                
                
                learn_start = time.time()
                for agent in self.agents:
                    self.algorithm.compute_returns(
                        last_critic_obs=critic_obs[agent],
                        agent_id=agent
                    )
              
            loss_dict = self.algorithm.update() #This will update both agent weights and critics
            learn_time = time.time() - learn_start
            self.current_learning_iteration = iteration

            # Log training progress
    def train_mode(self):
        """Switch to training mode"""
        self.policy.train()
        if self.normalize_observations:
            for agent in self.agents:
                self.actor_obs_normalizer[agent].train()
            for critic in self.critics:
                self.critic_obs_normalizer[critic].train()

    def eval_mode(self):
        """Switch to evaluation mode"""
        self.policy.eval()
        if self.normalize_observations:
            for agent in self.agents:
                self.actor_obs_normalizer[agent].eval()
            for critic in self.critics:
                self.critic_obs_normalizer[critic].eval()

    def save(self, path: str):
      pass

    def load(self, path: str):
      pass
    