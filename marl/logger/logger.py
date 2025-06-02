import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import statistics
from typing import Union, Callable, Dict
import numpy as np
import time
import sys

class Logger:
    """
    Logging tool with best stats tracking
    """
    
    def __init__(
        self, 
        log_dir: str = "logs",
        tensorboard = True,
        save_fn: Callable = None,
        verbose: bool = True):
        """
        A logger for logging data points as well as summary statistics.
        """
        self.tensorboard = tensorboard
        self.verbose = verbose

        self.tb_writer = None

        self.start_step = 0
        self.last_log_step = 0

        self.model_path = Path(log_dir) / "policy"
        self.video_path = Path(log_dir) / "videos"
        self.log_path = Path(log_dir) / "logs"

        self.log_path.mkdir(exist_ok=True, parents=True)
        self.model_path.mkdir(exist_ok=True, parents=True)
        self.video_path.mkdir(exist_ok=True, parents=True)

        if self.tensorboard:
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(str(self.log_path))

        self.save_fn = save_fn

        self.data = defaultdict(dict)
        self.data_log_summary = defaultdict(dict)
        self.stats = {}
        self.best_stats = {}
        self.save_fn = save_fn

        # Add timing and iteration tracking
        self.start_time = time.time()
        self.tot_timesteps = 0
        self.tot_time = 0
        self.iteration_time = 0
        self.last_iteration_time = 0
        self.current_iteration = 0
        self.start_iteration = 0
        self.num_learning_iterations = 0

        # self.rewbuffer = deque(maxlen=100) 
        # self.lenbuffer = deque(maxlen=100) 
        # self.episode_returns = []
        # self.episode_count = 0
        
        # # Initialize best stats tracking
        # self.best_stats = {
        #     'best_mean_reward': float('-inf'),
        #     'best_episode_return': float('-inf'),
        #     'best_iteration': 0,
        #     'best_model_saved': False
        # }
        
        # # Load existing best stats if they exist
        # self.load_best_stats()
    
    def store(self, tag="default", log_summary=False, **kwargs):
        """
        Stores scalar values or arrays into logger by tag and key to then be logged

        if log_summary is True, logs std, min, and max
        """
        for k, v in kwargs.items():
            self.data[tag][k] = v
            self.data_log_summary[tag][k] = log_summary
    
    def get_data(self, tag=None):
        if tag is None:
            data_dict = {}
            for tag in self.data.keys():
                for k, v in self.data[tag].items():
                    data_dict[f"{tag}/{k}"] = v
            return data_dict
        return self.data[tag]
    
    def log(self, step: int):
        log_string = {}
        
        self.last_log_step = step
        for tag in self.data.keys():
            data_dict = self.data[tag]
            for k, v in data_dict.items():
                key_vals = dict()
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    if len(v) > 0:
                        vals = np.array(v)
                        vals_sum, n = vals.sum(), len(vals)
                        avg = vals_sum / n
                        key_vals = {f"{tag}/{k}": avg}
                        if self.data_log_summary[tag][k]:
                            sum_sq = np.sum((vals - avg) ** 2)
                            std = np.sqrt(sum_sq / n)
                            minv = np.min(vals)
                            maxv = np.max(vals)
                            key_vals = {
                                **key_vals,
                                f"{tag}/{k}_std": std,
                                f"{tag}/{k}_min": minv,
                                f"{tag}/{k}_max": maxv,
                            }
                else:
                    key_vals = {f"{tag}/{k}": v}
                for name, scalar in key_vals.items():
                    if name not in self.best_stats:
                        update_val = True
                    else:
                        prev_val = self.best_stats[name]["val"]
                    if update_val:
                        self.best_stats[name] = dict(val=scalar, step=step)
                    if self.tensorboard:
                        self.tb_writer.add_scalar(name, scalar, self.start_step + step)
                    self.stats[name] = scalar

        log_string = self.create_log_string()
        if self.verbose:
            self.print(log_string)

        return self.stats

    def print(self, msg, file=sys.stdout):
        """
        print to terminal, stdout by default. Ensures only the main process ever prints.
        """
        print(msg, file=file)
        sys.stdout.flush()
    
    def log_iteration(self, iteration_num, timesteps_this_iter=None):
        """Call this at each iteration"""
        current_time = time.time()
        
        # Update iteration tracking
        self.current_iteration = iteration_num
        self.iteration_time = current_time - self.last_iteration_time 
        self.last_iteration_time = current_time
        
        # Update total time and timesteps
        self.tot_time = current_time - self.start_time
        if timesteps_this_iter:
            self.tot_timesteps += timesteps_this_iter

    def create_log_string(self, width=80, pad=35):
        """
        Pretty prints the logged statistics in a formatted way
        """
        log_string = f"{'#' * width}\n"
        
        # Get the data from stats
        data = self.get_data()
        
        # Print episode returns if available
        for metric in data: 
            if 'time' not in metric:
                log_string += f"{metric}: {data[metric]}\n"
        
        # Add timing and progress information
        log_string += f"{'-' * width}\n"
        log_string += f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        log_string += f"{'Iteration time:':>{pad}} {self.iteration_time:.2f}s\n"
        log_string += f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
        
        # Calculate ETA if we have all the necessary information
        if (self.current_iteration > self.start_iteration and 
            self.num_learning_iterations > 0):
            eta = self.tot_time / (self.current_iteration - self.start_iteration + 1) * (
                self.start_iteration + self.num_learning_iterations - self.current_iteration
            )
            log_string += f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"
        
        log_string += f"{'#' * width}\n"
        return log_string
        
        # collection_size = self.num_steps_per_env * self.env.num_envs
        # self.tot_timesteps += collection_size
        # self.tot_time += locs["collection_time"] + locs["learn_time"]
        # fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))
        # iteration = locs["it"]

        # # --- Losses
        # for key, value in locs["loss_dict"].items():
        #     self.tb_writer.add_scalar(f"Loss/{key}", value, iteration)
        #     if self.wandb_run:
        #         self.wandb_run.log({f"Loss/{key}": value}, step=iteration)

        # # --- Policy Info
        # mean_std = self.alg.policy.action_std.mean().item()
        # self.tb_writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)

        # # --- Performance Info
        # self.tb_writer.add_scalar("Perf/total_fps", fps, iteration)
        # self.tb_writer.add_scalar("Perf/collection time", locs["collection_time"], iteration)
        # self.tb_writer.add_scalar("Perf/learning_time", locs["learn_time"], iteration)

        # mean_rew = statistics.mean(self.episode_returns)
        # self.tb_writer.add_scalar("Train/mean_reward", mean_rew, iteration)
        # if self.logger_type != "wandb":
        #     self.tb_writer.add_scalar("Train/mean_reward/time", mean_rew, self.tot_time)

        # # Update best stats with current performance
        # current_stats = {
        #     'mean_reward': mean_rew,
        #     'fps': fps,
        #     'total_timesteps': self.tot_timesteps
        # }
        # self.update_best_stats(current_stats, iteration)

        # # --- Terminal Print (enhanced with best stats)
        # print_str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m \n"
        # print_str += f"{'#' * width}\n"
        # print_str += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
        # print_str += f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
        # for key, value in locs["loss_dict"].items():
        #     print_str += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"
        # if len(locs["rewbuffer"]) > 0:
        #     print_str += f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
        
        # # Add best stats to terminal output
        # print_str += f"{'Best mean reward:':>{pad}} {self.best_stats['best_mean_reward']:.2f} (iter {self.best_stats['best_iteration']})\n"
        # print_str += f"{'Best episode return:':>{pad}} {self.best_stats['best_episode_return']:.2f}\n"

        # print_str += "#" * width
        # print(print_str)
    
    def reset(self):
        """
        call this each time after log is called
        """
        self.data = defaultdict(dict)
        self.stats = {}
    
    def state_dict(self):
        return dict(best_stats=self.best_stats, last_log_step=self.last_log_step)

    def load(self, data):
        self.best_stats = data["best_stats"]
        self.last_log_step = data["last_log_step"]
        return self
    
   