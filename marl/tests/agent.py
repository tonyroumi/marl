import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import build_from_config

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def test_agent(cfg: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""
    env, agent = build_from_config(cfg)
    obs = env.reset()
    agent.learn()
    
    # Assert environment was created successfully
    assert env is not None
    
    env.close()
    print("Environment created successfully!")

if __name__ == "__main__":
    test_agent()