from typing import Dict, Tuple

import torch
import time
from collections import deque
import numpy as np

from marl.agents.base_marl import BaseMARLAgent


class BasicMARLAgent(BaseMARLAgent):
    """
    A basic single or multi-agent implementation of any algorithm extended to a multi-agent reinforcement 
    learning (MARL) setting via a shared or agent-specific actor-critic model structure.
    
    1. Support multiple independent actor-critic pairs (decentralized PPO).
    2. Maintain a shared actor-critic for all agents (centralized PPO).

    Can easily be extended to perform different updates for each agent, or even different algorithms for each agent.

    Core functionalities include:
    - Collecting trajectories by interacting with the environment.
    - Normalizing observations and preparing actions.
    - Updating the PPO policy using stored rollouts.
    - Maintaining episode statistics such as reward and length buffers.
    """
    

    def _collect_rollouts(
        self, 
        actor_obs: Dict, 
        critic_obs: Dict, 
        num_transitions: int
        ) -> Tuple[Dict, Dict, Dict]:
        """
        Collects rollout data from the environment for a specified number of transitions.

        Args:
            actor_obs: Current actor observations for all agents.
            critic_obs: Current critic observations for all agents.
            num_transitions: Number of environment steps to perform.

        Returns:
            The final observations after the rollout (actor and critic).
        """
        eps_ret = []
        for _ in range(num_transitions):
            actions = self.algorithm.act(actor_obs=actor_obs, critic_obs=critic_obs)
            actions = self.process_actions(actions)

            obs, rewards, dones, truncated, infos = self.env.step(actions)
            actor_obs, critic_obs = self.process_observations(obs)
            actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)
            
            if np.any(dones | truncated): #Log episode return when episode is complete
                eps_ret.append(infos.get("eps_ret"))
           
            self.algorithm.process_env_step(rewards, dones)

        self.logger.store(tag="episode", eps_ret=eps_ret, log_summary=True)

        return actor_obs, critic_obs

    def learn(self, total_iterations: int = 1000) -> None:
        """
        Main training loop for the PPO agent.

        Args:
            total_iterations (int): Number of learning iterations to perform.
        """
        self.train_mode()

        for iteration in range(total_iterations):
            iteration_start_time = time.time()

            obs, _ = self.env.reset()
            actor_obs, critic_obs = self.process_observations(obs)
            with torch.inference_mode():
                actor_obs, critic_obs = self._collect_rollouts(actor_obs, critic_obs, self.num_transitions_per_env)
                collection_time = time.time() - iteration_start_time
                self.algorithm.compute_returns(last_critic_obs=critic_obs)

            learn_start_time = time.time()
            loss_dict = self.algorithm.update()
            learn_time = time.time() - learn_start_time

            locs = {
                "collection_time": collection_time,
                "learn_time": learn_time,
                **loss_dict
            }

            self.logger.store(tag='training', log_summary=False, **locs)

            if iteration % self.save_interval == 0:
                self.logger.log(step=iteration)
                self.logger.reset()
                self.save(str(self.logger.model_path / f"{iteration}"))
        
            self.logger.log_iteration(iteration_num=iteration, timesteps_this_iter=self.num_transitions_per_env)
            
