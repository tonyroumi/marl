from typing import Dict, Tuple

import torch
import time
from collections import deque
import json
import os
import numpy as np # Make sure numpy is imported

from marl.agents.base_marl import BaseMARLAgent


class BasicMARLAgent(BaseMARLAgent):
    """
    A basic single or multi-agent implementation of any algorithm extended to a multi-agent reinforcement
    learning (MARL) setting via a shared or agent-specific actor-critic model structure.
    (Rest of the docstring as before)
    """

    def __init__(self, env, policy, algorithm, observation_config, num_transitions_per_env, normalize_observations, logger=None, device=None):
        super().__init__(env, policy, algorithm, observation_config, num_transitions_per_env, normalize_observations, logger, device)
        self.rewbuffer = deque(maxlen=100) 
        self.lenbuffer = deque(maxlen=100) 

    def _collect_rollouts(
        self,
        actor_obs: Dict,
        critic_obs: Dict,
        num_transitions: int
        ) -> Tuple[Dict, Dict]:
        """
        Collects rollout data from the environment for a specified number of transitions.
        """
        for step_idx in range(num_transitions):
            actions = self.algorithm.act(actor_obs=actor_obs, critic_obs=critic_obs)
            actions = self.process_actions(actions)

            obs, rewards, dones, truncated, step_infos_from_env = self.env.step(actions)
            # print(f"Step {step_idx} infos: {step_infos_from_env}") # For debugging info content

            current_infos = step_infos_from_env
            if not isinstance(current_infos, list): # If it's a single env, wrap info in a list
                current_infos = [current_infos]
            
            current_dones = dones
            # Ensure dones is list-like for consistent indexing, especially for single env
            if not isinstance(current_dones, (list, np.ndarray)): 
                current_dones = np.array([current_dones]) if isinstance(current_dones, (bool, np.bool_)) else np.array(current_dones)
            
            current_truncated = truncated
            # Ensure truncated is list-like for consistent indexing
            if not isinstance(current_truncated, (list, np.ndarray)):
                current_truncated = np.array([current_truncated]) if isinstance(current_truncated, (bool, np.bool_)) else np.array(current_truncated)


            for i in range(len(current_infos)):
                info_item = current_infos[i]
                if info_item is None: 
                    continue

                # Check if episode ended for this specific sub-environment
                # Note: dones and truncated should be arrays/lists of booleans here
                episode_ended = False
                if i < len(current_dones) and current_dones[i]:
                    episode_ended = True
                if i < len(current_truncated) and current_truncated[i]:
                    episode_ended = True
                
                if episode_ended:
                    episode_return_value = None
                    episode_length_value = None

                    if "episode" in info_item and isinstance(info_item["episode"], dict):
                        episode_return_value = info_item["episode"].get("r")
                        episode_length_value = info_item["episode"].get("l")
                    elif 'eps_ret' in info_item and 'eps_len' in info_item:
                        episode_return_value = info_item['eps_ret']
                        episode_length_value = info_item['eps_len']
                    # else: # Useful for debugging what info keys are actually present
                        # print(f"Debug: Env {i} episode ended. Keys in info_item: {info_item.keys()}")


                    if episode_return_value is not None:
                        try:
                            scalar_val = 0.0 # Default
                            if isinstance(episode_return_value, (list, np.ndarray)):
                                # Convert to numpy array for consistent processing
                                ep_ret_array = np.asarray(episode_return_value, dtype=float)
                                if ep_ret_array.size == 0: 
                                    scalar_val = 0.0
                                elif ep_ret_array.ndim == 0: # Already a scalar in a 0-d array
                                    scalar_val = float(ep_ret_array.item())
                                else: # 1-d or higher, take the mean
                                    scalar_val = float(np.mean(ep_ret_array))
                            elif isinstance(episode_return_value, (int, float, np.number)):
                                scalar_val = float(episode_return_value)
                            elif hasattr(episode_return_value, 'item'): # For 0-dim tensors/numpy scalars not caught by np.number
                                scalar_val = float(episode_return_value.item())
                            else:
                                print(f"Warning: Unhandled type for episode_return_value: {type(episode_return_value)}, value: {episode_return_value}. Appending 0.0.")
                                scalar_val = 0.0
                            self.rewbuffer.append(scalar_val)
                        except Exception as e:
                            print(f"Warning: Could not process episode_return_value: '{episode_return_value}' (type: {type(episode_return_value)}). Error: {e}. Appending 0.0.")
                            self.rewbuffer.append(0.0) # Append a default value

                    if episode_length_value is not None:
                        try:
                            scalar_val = 0.0 # Default
                            if isinstance(episode_length_value, (list, np.ndarray)):
                                ep_len_array = np.asarray(episode_length_value, dtype=float)
                                if ep_len_array.size == 0:
                                     scalar_val = 0.0
                                elif ep_len_array.ndim == 0:
                                     scalar_val = float(ep_len_array.item())
                                else:
                                    scalar_val = float(np.mean(ep_len_array)) # Or sum, or first element, depending on meaning
                            elif isinstance(episode_length_value, (int, float, np.number)):
                                scalar_val = float(episode_length_value)
                            elif hasattr(episode_length_value, 'item'):
                                scalar_val = float(episode_length_value.item())
                            else:
                                print(f"Warning: Unhandled type for episode_length_value: {type(episode_length_value)}, value: {episode_length_value}. Appending 0.0.")
                                scalar_val = 0.0
                            self.lenbuffer.append(scalar_val)
                        except Exception as e:
                            print(f"Warning: Could not process episode_length_value: '{episode_length_value}' (type: {type(episode_length_value)}). Error: {e}. Appending 0.0.")
                            self.lenbuffer.append(0.0) # Append a default value
            
            actor_obs, critic_obs = self.process_observations(obs)
            actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)
            self.algorithm.process_env_step(rewards, dones)

        return actor_obs, critic_obs

    def learn(self, total_iterations: int = 10) -> None:
        obs, info = self.env.reset() 
        actor_obs, critic_obs = self.process_observations(obs)
        self.train_mode()

        training_log_data = []

        def make_serializable(data):
            if isinstance(data, torch.Tensor):
                data_cpu = data.detach().cpu()
                if data_cpu.numel() == 1:
                    val = data_cpu.item()
                    if np.isnan(val): return "NaN"
                    if np.isinf(val): return "Infinity" if val > 0 else "-Infinity"
                    return val
                else: 
                    return [make_serializable(x) for x in data_cpu.tolist()]
            elif isinstance(data, dict):
                return {k: make_serializable(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [make_serializable(i) for i in data]
            elif isinstance(data, np.ndarray):
                if data.ndim == 0: 
                    val = data.item()
                    if np.isnan(val): return "NaN"
                    if np.isinf(val): return "Infinity" if val > 0 else "-Infinity"
                    return val
                return [make_serializable(x) for x in data.tolist()] 
            elif isinstance(data, (np.generic)): 
                val = data.item() # This should handle np.bool_ as well
                if isinstance(val, bool): return val # Ensure bools are not Nan/Inf checked
                if np.isnan(val): return "NaN"
                if np.isinf(val): return "Infinity" if val > 0 else "-Infinity"
                return val
            elif isinstance(data, float): 
                if np.isnan(data): return "NaN"
                if np.isinf(data): return "Infinity" if data > 0 else "-Infinity"
                return data
            elif isinstance(data, (int, str, bool)) or data is None:
                return data
            
            print(f"Warning: Type {type(data)} with value {data} not specifically handled by make_serializable, converting to string.")
            return str(data)


        print(f"Starting training for {total_iterations} iterations...")

        for iteration in range(total_iterations):
            iteration_start_time = time.time() 
            current_actor_obs = actor_obs
            current_critic_obs = critic_obs

            with torch.inference_mode():
                next_actor_obs, next_critic_obs = self._collect_rollouts(current_actor_obs, current_critic_obs, self.num_transitions_per_env)
                collection_time = time.time() - iteration_start_time
                self.algorithm.compute_returns(last_critic_obs=next_critic_obs)
                actor_obs = next_actor_obs
                critic_obs = next_critic_obs

            learn_start_time = time.time()
            loss_dict = self.algorithm.update()
            learn_time = time.time() - learn_start_time

            log_entry = {
                "iteration": iteration + 1,
                "collection_time_seconds": round(collection_time, 4),
                "learn_time_seconds": round(learn_time, 4),
                "losses": make_serializable(loss_dict)
            }

            if len(self.rewbuffer) > 0:
                try:
                    # Buffer should contain floats after processing in _collect_rollouts
                    # Filter out any non-numeric items just in case, though ideally not needed if _collect_rollouts is perfect
                    numeric_rewards = [r for r in list(self.rewbuffer) if isinstance(r, (int,float)) and not (np.isnan(r) or np.isinf(r))]
                    if numeric_rewards: 
                        log_entry["average_episode_return"] = round(float(torch.mean(torch.tensor(numeric_rewards, dtype=torch.float))), 4)
                    else:
                        log_entry["average_episode_return"] = "N/A (no valid rewards in buffer)"
                except Exception as e:
                    print(f"Warning: Could not calculate/serialize rewbuffer mean for JSON (iteration {iteration + 1}): {e}")
                    log_entry["average_episode_return"] = "N/A (error calculating mean)"
            else:
                log_entry["average_episode_return"] = "N/A (buffer empty)"
            
            if len(self.lenbuffer) > 0:
                try:
                    numeric_lengths = [l for l in list(self.lenbuffer) if isinstance(l, (int,float)) and not (np.isnan(l) or np.isinf(l))]
                    if numeric_lengths:
                         log_entry["average_episode_length"] = round(float(torch.mean(torch.tensor(numeric_lengths, dtype=torch.float))), 2)
                    else:
                        log_entry["average_episode_length"] = "N/A (no valid lengths in buffer)"
                except Exception as e:
                    print(f"Warning: Could not calculate/serialize lenbuffer mean for JSON (iteration {iteration + 1}): {e}")
                    log_entry["average_episode_length"] = "N/A (error calculating mean)"
            else:
                log_entry["average_episode_length"] = "N/A (buffer empty)"


            training_log_data.append(log_entry)

            if (iteration + 1) % 10 == 0: 
                 print(f"Iteration {iteration + 1}/{total_iterations} completed. Avg Return (last 100): {log_entry.get('average_episode_return', 'N/A')}")

            self.current_learning_iteration = iteration
        
        log_dir = "training_logs_json"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(log_dir, f"marl_training_log_{timestamp}.json")

        print(f"Attempting to save training log to {log_file_path}...")
        
        try:
            with open(log_file_path, 'w') as f:
                json.dump(training_log_data, f, indent=4)
            print(f"Training log successfully saved to {log_file_path}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to save JSON log to {log_file_path}. Error: {e}")
            print("Data that could not be saved (first 5 entries if available):")
            for i, entry in enumerate(training_log_data[:5]):
                print(f"Entry {i}: {entry}")
                try:
                    json.dumps(entry) # Test individual entry
                except Exception as entry_e:
                    print(f"  Problem serializing this entry ({type(entry)}): {entry_e}")
                    # If an entry is problematic, try to print its problematic parts
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            try:
                                json.dumps({k: v})
                            except Exception as item_e:
                                print(f"    Problem with item {k}: {v} (type: {type(v)}), error: {item_e}")


        print("Training finished.")