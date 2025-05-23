from marl.policies.base_policy import BasePolicy
from typing import Any, Dict
import torch
import time
from collections import deque
from marl.algorithms.base import BaseAlgorithm
import numpy as np
from marl.networks import EmpiricalNormalization

class BasicAgent:
  def __init__(
      self, 
      env: Any,
      policy: BasePolicy, 
      algorithm: BaseAlgorithm,
      actor_obs_keys: Dict[str, Any],
      critic_obs_keys: Dict[str, Any],
      normalize_observations: bool,
      preprocess_observations: bool,
      logger: Any = None,
      ):
    self.device = policy.device

    self.env = env
    self.env.num_envs = 1 #Placeholder. add vectorized envs later
    self.action_dim = self.env.action_dim
    
    self.policy = policy
    self.algorithm = algorithm
    self.logger = logger

    self.actor_obs_keys = actor_obs_keys
    self.critic_obs_keys = critic_obs_keys

    self.normalize_observations = normalize_observations
    self.preprocess_observations = preprocess_observations

    self._post_init()

  def _post_init(self):
    obs = self.env._get_observations()

    # Compute observation dimensions for each agent and critic
    self.num_actor_obs = {}
    self.num_critic_obs = {}
    
    for agent in self.actor_obs_keys:
        # Sum up the dimensions of all observation components for this agent
        self.num_actor_obs[agent] = sum(obs[key].shape[-1] for key in self.actor_obs_keys[agent])
    
    for critic in self.critic_obs_keys:
        # Sum up the dimensions of all observation components for this critic
        self.num_critic_obs[critic] = sum(obs[key].shape[-1] for key in self.critic_obs_keys[critic])

    self.agents = list(self.actor_obs_keys.keys())
    self.critics = list(self.critic_obs_keys.keys())

    self.actor_obs_normalizer = {}
    self.critic_obs_normalizer = {}

    if self.normalize_observations:
      for agent in self.agents:
        self.actor_obs_normalizer[agent] = EmpiricalNormalization(shape=[self.num_actor_obs[agent]], until=1.0e8).to(self.device)
      for critic in self.critics:
        self.critic_obs_normalizer[critic] = EmpiricalNormalization(shape=[self.num_critic_obs[critic]], until=1.0e8).to(self.device)
    else:
      self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
      self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

    self.buffers = {}
    self.initialize_buffers()

  def initialize_buffers(self):
    for agent in self.actor_obs_keys:
      self.buffers[agent] = self.algorithm._init_storage(
        self.env.num_envs,
        agent
      )
    
  def learn(self):
    #initalize logger bla bla
    all_obs = self.env._get_observations()
    actor_obs, critic_obs = self.process_observations(all_obs)

    ep_infos = []
    rewbuffer = deque(maxlen=100) #maybe we can have a wrapper to save this instead
    lenbuffer = deque(maxlen=100)

    start_iter = 0
    tot_iter = 3 #TODO

    for it in range(start_iter, tot_iter):
      start = time.time()
      #Collect rollouts
      actions = np.zeros((len(self.agents), self.action_dim))  
      with torch.inference_mode():
        for _ in range(self.algorithm.num_steps_per_env['agent_0']):
          for i, agent_id in enumerate(self.agents):
            actions[i] = self.algorithm.act(
              actor_obs=actor_obs[agent_id],
              critic_obs=critic_obs[agent_id],
              agent_id=agent_id
            ).cpu().numpy()  

          obs, rewards, dones, infos = self.env.step(actions.squeeze())

          actor_obs, critic_obs = self.process_observations(obs)

          for agent in self.agents:
            actor_obs[agent] = self.actor_obs_normalizer[agent](actor_obs[agent])
          
          for critic in self.critics:
            critic_obs[critic] = self.critic_obs_normalizer[critic](critic_obs[critic])

          for agent in self.agents:
            self.algorithm.process_env_step(
              rewards,
              dones,
              agent
            )

            # logger.log()
        stop = time.time()
        collection_time = stop - start
        start = stop

        for agent in self.agents:
          self.algorithm.compute_returns(
            last_critic_obs=critic_obs[agent],
            agent_id=agent
            )
      loss_dict = self.algorithm.update('agent_0') #handle each agent appropriately
      stop = time.time()
      learn_time = stop - start



  def save(self, path: str):
    #self.policy.save(path)
    pass

  def load(self, path: str):
    #self.policy.load(path)
    pass

  def process_observations(self, all_obs):
    """
    Process observations by selecting and concatenating relevant observations for actors and critics.
    
    Args:
        all_obs (Dict): Dictionary containing all observations from the environment
        
    Returns:
        Tuple[Dict, Dict]: Tuple containing dictionaries of processed actor and critic observations
    """
    actor_obs = {}
    critic_obs = {}
    
    # Process actor observations
    for agent in self.agents:
        # Select and concatenate observations for this agent
        agent_obs = torch.cat([all_obs[key] for key in self.actor_obs_keys[agent]], dim=-1).to(self.device)
        actor_obs[agent] = agent_obs
    
    # Process critic observations
    for critic in self.critics:
        # Select and concatenate observations for this critic
        critic_obs_tensor = torch.cat([all_obs[key] for key in self.critic_obs_keys[critic]], dim=-1).to(self.device)
        critic_obs[critic] = critic_obs_tensor
    
    return actor_obs, critic_obs
  