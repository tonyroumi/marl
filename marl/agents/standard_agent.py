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
      actor_obs_keys: Dict[str, Any],
      critic_obs_keys: Dict[str, Any],
      normalize_observations: bool,
      preprocess_observations: bool,
      ):
    super().__init__(policy)
    self.device = policy.device

    self.env = env
    self.env.num_envs = 1 #Placeholder. add vectorized envs later
    
    self.policy = policy
    self.algorithm = algorithm
    
    self.actor_obs_keys = actor_obs_keys
    self.critic_obs_keys = critic_obs_keys

    self.agents = list(self.actor_obs_keys.keys())
    self.critics = list(self.critic_obs_keys.keys())

    self.normalize_observations = normalize_observations
    self.preprocess_observations = preprocess_observations

    self.buffers = {}
    self.initialize_buffers()

  def initialize_buffers(self):
    for agent in self.actor_obs_keys:
      self.buffers[agent] = self.algorithm._init_storage(
        self.env.num_envs,
        agent
      )
  
  def preprocess_obs(self, obs):
    pass

  def preprocess_img_obs(self, obs):
    pass

  def act(self, obs, privileged_obs):
    pass

  def learn(self):
    #initalize logger bla bla
    all_obs = self.env._get_observations()

    actor_obs, critic_obs = {}, {}
    for agent, critic in zip(self.agents, self.critics):
      actor_obs[agent] = torch.cat([all_obs[key] for key in self.actor_obs_keys[agent]], dim=-1).to(self.device)
      critic_obs[critic] = torch.cat([all_obs[key] for key in self.critic_obs_keys[critic]], dim=-1).to(self.device)

    ep_infos = []
    rewbuffer = deque(maxlen=100) #maybe we can have a wrapper to save this instead
    lenbuffer = deque(maxlen=100)

    start_iter = 0
    tot_iter = 1000000 #TODO

    for it in range(start_iter, tot_iter):
      start = time.time()
      #Collect rollouts
      with torch.inference_mode():
        for _ in range(self.algorithm.num_steps_per_env['agent_0']):
          actions, info = self.policy.act(actor_obs) #RESUME HERE
          concatenated_actions = torch.cat([actions[agent] for agent in self.agents], dim=-1)
          obs, rewards, dones, infos = self.env.step(concatenated_actions.to(self.device))
          

          obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

          obs = self.obs_normalizer(obs)

          if self.privileged_obs_dim is not None:
            privileged_obs = self.privileged_obs_normalizer(
              privileged_obs,
            )
          else:
            privileged_obs = obs
          
          self.algorithm.process_env_step( #This may need to be different depending on the agent
            rewards,
            dones,
            infos
          )

            # logger.log()
        stop = time.time()
        collection_time = stop - start
        start = stop

        self.algorithm.compute_returns(privileged_obs) #not sure how to handle this per agent
      loss_dict = self.algorithm.update() #handle each agent appropriately
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      self.logger.log(
        #puthere
      )
      # logger.log()

      # self.algorithm.update_normalizers(infos)

      # self.algorithm.update_target_networks()

          



  def save(self, path: str):
    #self.policy.save(path)
    pass

  def load(self, path: str):
    #self.policy.load(path)
    pass
  