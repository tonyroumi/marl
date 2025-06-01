from typing import Dict, Tuple

import torch
import time
from collections import deque

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
        ) -> Tuple[Dict, Dict]:
        """
        Collects rollout data from the environment for a specified number of transitions.

        Args:
            actor_obs: Current actor observations for all agents.
            critic_obs: Current critic observations for all agents.
            num_transitions: Number of environment steps to perform.

        Returns:
            The final observations after the rollout (actor and critic).
        """
        for _ in range(num_transitions):
            actions = self.algorithm.act(actor_obs=actor_obs, critic_obs=critic_obs)
            actions = self.process_actions(actions)

            obs, rewards, dones, truncated, infos = self.env.step(actions)
            actor_obs, critic_obs = self.process_observations(obs)
            actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)

           
            self.algorithm.process_env_step(rewards, dones)

        return actor_obs, critic_obs

    def learn(self, total_iterations: int = 1000) -> None:
        """
        Main training loop for the PPO agent.

        Args:
            total_iterations (int): Number of learning iterations to perform.
        """
        obs, info = self.env.reset()
        actor_obs, critic_obs = self.process_observations(obs)
        print(actor_obs, critic_obs)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        curr_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        curr_ep_len = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        for iteration in range(total_iterations):
            start_time = time.time()

            with torch.inference_mode():
                actor_obs, critic_obs = self._collect_rollouts(actor_obs, critic_obs, self.num_transitions_per_env)
                collection_time = time.time() - start_time
              
                self.algorithm.compute_returns(last_critic_obs=critic_obs)

                learn_start = time.time()


            loss_dict = self.algorithm.update()
            learn_time = time.time() - learn_start

            self.current_learning_iteration = iteration