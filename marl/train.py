import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import instantiate_all
from marl.utils.utils import set_seed

@hydra.main(version_base=None, config_path=".configs", config_name="config")
def train(config: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""
    set_seed(config.seed)

    env, policy, algorithm, agent = instantiate_all(config)

    env.reset()
    agent.learn()
    
    env.close()

if __name__ == "__main__":
    train()