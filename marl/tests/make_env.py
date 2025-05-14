import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import build_env_from_config


@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def test_build_env_from_config(cfg: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""

    env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    env = build_env_from_config(env_config)
    
    # Assert environment was created successfully
    assert env is not None
    
    env.close()
    print("Environment created successfully!")

if __name__ == "__main__":
    test_build_env_from_config()