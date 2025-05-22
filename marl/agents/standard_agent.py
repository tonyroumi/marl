from marl.agents.base_agent import BaseAgent
from marl.policies.base_policy import BasePolicy
from marl.algorithms.base_algorithm import BaseAlgorithm
from typing import Any, Dict
import torch
import time
from collections import deque

class StandardAgent(BaseAgent):
  def __init__(
      self, 
      env: Any,
      policy: BasePolicy, 
      algorithm: BaseAlgorithm,
      agent_cfg: Dict[str, Any],
      algorithm_cfg: Dict[str, Any],
      **kwargs: Any):
    super().__init__(policy, **kwargs)
    self.device = policy.device

    self.env = env
    self.policy = policy
    self.algorithm = algorithm
    
    self.algorithm_cfg = algorithm_cfg
    self.agent_cfg = agent_cfg

    self.agents = agent_cfg.get("agents", [])
    self.critics = agent_cfg.get("critic", self.agents)

    self.buffers = {}

    # self.logger = logger()

    self.initialize_buffers()

  def initialize_buffers(self):
    for agent, critic in zip(self.agents, self.critics):
      self.buffers[agent] = self.algorithm.init_storage(
        1,
        self.algorithm_cfg.get("num_transitions_per_env", 1000),
        self.policy.components[agent].network_kwargs['actor_obs_dim'],
        self.policy.components[critic].network_kwargs['critic_obs_dim'],
        self.agent_cfg.get("num_actions", 14),
        self.device
      )
  
  def preprocess_obs(self, obs):
    pass


  def act(self, obs, privileged_obs):
    pass

  def learn(self):
    #initalize logger bla bla
    obs, info = self.env.get_obs()
    privileged_obs = obs

    obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)

    ep_infos = []
    rewbuffer = deque(maxlen=100) #maybe we can have a wrapper to save this instead
    lenbuffer = deque(maxlen=100)

    start_iter = 0
    tot_iter = self.algorithm_cfg.get("total_timesteps", 1000000)

    for it in range(start_iter, tot_iter):
      start = time.time()
      if it < stat_iter:
        self.policy.set_train_mode()
        self.algorithm.set_train_mode()
      else:
        self.policy.set_eval_mode()
        self.algorithm.set_eval_mode()

    pass

  


  