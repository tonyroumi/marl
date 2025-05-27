import gymnasium as gym


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Adds additional info. Anything that goes in the stats wrapper is logged to tensorboard/wandb under train_stats and test_stats
    """

    def reset(self, *, seed=None, options=None):
        self.eps_seed = seed
        obs, info = super().reset(seed=seed, options=options)
        self.eps_ret = 0
        self.eps_len = 0
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["seed"] = self.eps_seed
        if "episode" in info:
            info["eps_ret"] = info["stats"]["return"]
            info["eps_len"] = info["stats"]["episode_len"]
        else:
            self.eps_ret += reward
            self.eps_len += 1
            info["eps_ret"] = self.eps_ret
            info["eps_len"] = self.eps_len
        return observation, reward, terminated, truncated, info


# class RecordVideoWrapper(Wrapper):
#     """
#     A wrapper that records a video of the environment for all cameras in the environment.
#     """
    
#     def __init__(self, env):
#         super().__init__(env)
#         self.frames = {camera: [] for camera in self.env.camera_names}

#     def reset(self):
#         obs = super().reset()
        
#         # Clear previous frames
#         self.frames = {camera: [] for camera in self.env.camera_names}
                       
#         return obs
        
#     def step(self, action):
#         obs, reward, done, info = super().step(action)
        
#         # Collect frames from each camera
#         for camera in self.env.camera_names:
#             self.frames[camera].append(obs[camera + "_image"])
                
#         return obs, reward, done, info
     
#     def save_videos(self):
#         """ Save the recorded frames as videos, one for each camera. """
        
#         # For each camera, save a separate video
#         for camera, frames in self.frames.items():
#             if not frames:  # Skip if no frames were recorded
#                 continue
                
#             # Create filename with camera name
#             camera_filename = f"{camera}.mp4"
            
#             # Create a writer for this camera
#             writer = imageio.get_writer(camera_filename, fps=20)
            
#             # Write all frames for this camera
#             for frame in frames:
#                 writer.append_data(frame)
                
#             # Close the writer
#             writer.close()
