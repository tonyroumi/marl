import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import instantiate_all
from marl.utils.utils import set_seed
from hydra.core.hydra_config import HydraConfig
import os

@hydra.main(version_base=None, config_path=".configs", config_name="config")
def train(config: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""
    set_seed(config.seed)
    total_timesteps = config.total_timesteps
    
    # Get Hydra's output directory
    save_dir = HydraConfig.get().runtime.output_dir

    env, policy, algorithm, agent = instantiate_all(config)

    env.reset()

    try:
        agent.learn(total_timesteps)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        agent.save(os.path.join(save_dir, "policy"))
    finally:
        env.close()
        agent.save(os.path.join(save_dir, "policy"))

if __name__ == "__main__":
    train()