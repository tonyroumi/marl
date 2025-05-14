# Run this with the following command to not save to results folder
# ''' python inspector.py hydra.run.dir=. hydra.output_subdir=null  '''

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from PIL import Image

from marl.utils.config_utils import build_env_from_config, build_policy_from_config


@hydra.main(config_path="../.configs", config_name="config", version_base=None)
def inspect(cfg: DictConfig):
    """
    Analyze environment observations, model sizes, and save images.
    """
    cfg =  OmegaConf.to_container(cfg, resolve=True)
    
    env = build_env_from_config(cfg['environment'])
    
    # Reset environment to get initial observation
    obs = env.reset()
    robot_obs_counts = {}
    total_obs_count = 0
    rgb_image_count = 0
    depth_image_count = 0

    print("\n=== Observation Keys and Shapes ===")
    for key, value in obs.items():
        total_obs_count += 1
        
        # Print observation shape/type as before
        if isinstance(value, (np.ndarray, list)):
            shape = np.array(value).shape
            print(f"{key}: {shape}")
        else:
            print(f"{key}: {type(value)}")
    
    
    print("\n=== Model Sizes ===")

    policy = build_policy_from_config(cfg['policies'])
    
   
    for name, component in policy.components.items():
        print(f"Policy component: {name}")
        # If component has parameters (like neural networks)
        
        params = component.parameters()
        for agent_id, params in params.items():
            total_params = sum(p.numel() for p in params)
            print(f"  Agent {agent_id},  Parameters: {total_params:,}")
    
    print("\n=== Saving Images ===")
    
    # Save any observation that has "_image" in the key
    for key, value in obs.items():
        if "_image" in key or "_depth" in key and isinstance(value, np.ndarray):
            print(f"Saving image from observation: {key}")
            # Convert to correct format if needed
            if value.dtype == np.float32:
                if value.max() <= 1.0:
                    value = (value * 255).astype(np.uint8)
            
            if len(value.shape) == 3:
                if value.shape[2] == 3:  # RGB format
                    img = Image.fromarray(value)
                    img.save(f"{key}.png")
                else:
                    # Handle other channel configurations
                    plt.figure(figsize=(10, 10))
                    # Use a colormap suitable for depth (viridis, jet, etc)
                    im = plt.imshow(value, cmap='viridis')
                    # Add colorbar with labels
                    cbar = plt.colorbar(im)
                    cbar.set_label('Depth (closer ← → further)')
                    plt.savefig(f"{key}.png")
                    plt.close()
            else:
                # Handle grayscale or unusual dimensions
                plt.figure(figsize=(10, 10))
                plt.imshow(value, cmap='gray')
                plt.savefig(f"{key}.png")
                plt.close()
    
    # Close the environment
    env.close()
    print("\nInspection complete!")


if __name__ == "__main__":
    inspect()
