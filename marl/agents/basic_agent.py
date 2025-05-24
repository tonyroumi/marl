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

      self.device = device

      self.actor_obs_keys = self.observation_config['actor_obs_keys']
      self.critic_obs_keys = self.observation_config['critic_obs_keys'] \
        if self.observation_config['critic_obs_keys'] else self.actor_obs_keys

      self._post_init()

    def _post_init(self):
        """Initialize derived properties after main initialization."""
        obs = self.env._get_observations()
        
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
            
            for agent in self.agents:
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
        self.buffers = self.algorithm._init_storage(self.env.num_envs)
    
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
    
    def _collect_rollouts(self, actor_obs: Dict, critic_obs: Dict) -> Tuple[Dict, Dict]:
        """Collect rollout experiences from environment interactions."""
        actions = np.zeros((len(self.agents), self.env.action_dim // len(self.agents)))
        
        with torch.inference_mode():
            for _ in range(self.algorithm.num_steps_per_env['agent_0']): #How should I go about this ?
                # Get actions from all agents
                for i, agent_id in enumerate(self.agents):
                    with torch.inference_mode():  
                        actions[i] = self.algorithm.act(
                            actor_obs=actor_obs[agent_id],
                            critic_obs=critic_obs[agent_id],
                            agent_id=agent_id
                        ).cpu().numpy()

                # Step environment
                obs, rewards, dones, infos = self.env.step(actions.squeeze())
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
        # Initialize tracking
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        
        # Get initial observations
        all_obs = self.env._get_observations()
        actor_obs, critic_obs = self.process_observations(all_obs)
        
        for iteration in range(total_iterations):
            start_time = time.time()
            
            # Collect rollouts
            actor_obs, critic_obs = self._collect_rollouts(actor_obs, critic_obs)
            collection_time = time.time() - start_time
            
            # Compute returns and update
            with torch.inference_mode():
                learn_start = time.time()
                for agent in self.agents:
                  self.algorithm.compute_returns(
                      last_critic_obs=critic_obs[agent],
                      agent_id=agent
                  )
              
            # Update algorithm (currently handles only first agent)

            loss_dict = self.algorithm.update() #This well update both agent weights and critics
            learn_time = time.time() - learn_start

            # Log training progress

    def save(self, path: str):
      pass

    def load(self, path: str):
      pass
    