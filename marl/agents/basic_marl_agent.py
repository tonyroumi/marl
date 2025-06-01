from typing import Dict, Tuple

import numpy as np
import torch
import time
from collections import deque
import json # Added for JSON logging
import os   # Added for path creation

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
        for step_idx in range(num_transitions):
            actions = self.algorithm.act(actor_obs=actor_obs, critic_obs=critic_obs)
            actions = self.process_actions(actions)

            obs, rewards, dones, truncated, infos = self.env.step(actions)
            print(infos) # This print remains as per "don't touch anything else"

            actor_obs, critic_obs = self.process_observations(obs)
            actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)


            self.algorithm.process_env_step(rewards, dones)

        return actor_obs, critic_obs

    def learn(self, total_iterations: int = 10) -> None:
        """
        Main training loop for the PPO agent.

        Args:
            total_iterations (int): Number of learning iterations to perform.
        """
        obs, info = self.env.reset()
        actor_obs, critic_obs = self.process_observations(obs)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        curr_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        curr_ep_len = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        # --- Initialize list to store log data ---
        training_log_data = []

        # --- Helper function to make data JSON serializable ---
        def make_serializable(data):
            if isinstance(data, torch.Tensor):
                # Detach tensor from graph, move to CPU, convert to Python list/number
                data = data.detach().cpu()
                return data.item() if data.numel() == 1 else data.tolist()
            if isinstance(data, dict):
                return {k: make_serializable(v) for k, v in data.items()}
            if isinstance(data, list):
                return [make_serializable(i) for i in data]
            if isinstance(data, np.ndarray):
                return data.tolist()
            if hasattr(data, 'item'): # For numpy scalars
                return data.item()
            return data

        print(f"Starting training for {total_iterations} iterations...")

        for iteration in range(total_iterations):
            start_time = time.time()

            # --- Store original actor_obs and critic_obs for this iteration ---
            current_actor_obs = actor_obs
            current_critic_obs = critic_obs

            with torch.inference_mode():
                # --- Pass the current obs to _collect_rollouts ---
                next_actor_obs, next_critic_obs = self._collect_rollouts(current_actor_obs, current_critic_obs, self.num_transitions_per_env)
                collection_time = time.time() - start_time

                # --- Use the obs returned by _collect_rollouts for compute_returns ---
                self.algorithm.compute_returns(last_critic_obs=next_critic_obs)

                # --- Update actor_obs and critic_obs for the next iteration ---
                actor_obs = next_actor_obs
                critic_obs = next_critic_obs

            learn_start_time = time.time()
            loss_dict = self.algorithm.update()
            learn_time = time.time() - learn_start_time

            # --- Create log entry for the current iteration ---
            log_entry = {
                "iteration": iteration + 1,
                "collection_time_seconds": round(collection_time, 4),
                "learn_time_seconds": round(learn_time, 4),
                "losses": make_serializable(loss_dict) # Ensure losses are serializable
            }

            # --- (Optional) Add mean reward and episode length from buffers if they are populated ---
            # Note: Ensure rewbuffer and lenbuffer are populated correctly by your environment wrappers
            if len(rewbuffer) > 0:
                try:
                    log_entry["mean_reward_buffer"] = make_serializable(torch.mean(torch.tensor([make_serializable(item) for item in list(rewbuffer)], dtype=torch.float)))
                except Exception as e:
                    print(f"Warning: Could not serialize rewbuffer mean: {e}") # or log this warning
            if len(lenbuffer) > 0:
                try:
                    log_entry["mean_episode_length_buffer"] = make_serializable(torch.mean(torch.tensor([make_serializable(item) for item in list(lenbuffer)], dtype=torch.float)))
                except Exception as e:
                    print(f"Warning: Could not serialize lenbuffer mean: {e}") # or log this warning


            training_log_data.append(log_entry)

            # --- Minimal console output, primary log is JSON ---
            if (iteration + 1) % 10 == 0: # Print progress every 10 iterations
                 print(f"Iteration {iteration + 1}/{total_iterations} completed. Collection: {collection_time:.2f}s, Learn: {learn_time:.2f}s")
                 # print(f"Losses: {make_serializable(loss_dict)}")


            self.current_learning_iteration = iteration

        # --- Save the collected log data to a JSON file ---
        log_dir = "training_logs_json"
        os.makedirs(log_dir, exist_ok=True)
        # You can customize the filename, e.g., based on experiment name or timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(log_dir, f"marl_training_log_{timestamp}.json")

        try:
            with open(log_file_path, 'w') as f:
                json.dump(training_log_data, f, indent=4)
            print(f"Training log saved to {log_file_path}")
        except Exception as e:
            print(f"Error saving JSON log: {e}")
            # Fallback or alternative saving if needed
            # For example, print the problematic data:
            # for i, entry in enumerate(training_log_data):
            #     try:
            #         json.dumps(entry)
            #     except TypeError:
            #         print(f"Problematic entry at index {i}: {entry}")


        print("Training finished.")