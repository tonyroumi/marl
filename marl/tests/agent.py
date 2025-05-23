import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import build_env_from_config, build_policy_from_config, build_agent_from_config, parse_agent_configs
from marl.agents.standard_agent import StandardAgent
from marl.algorithms.ppo import PPO
from marl.envs.wrappers.common import TorchObsWrapper

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def test_agent(cfg: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    agent_configs = parse_agent_configs(cfg['algorithm'])
    env, agent = build_agent_from_config(cfg)
    obs = env.reset()
    agent.learn()
    
    # Assert environment was created successfully
    assert env is not None
    
    env.close()
    print("Environment created successfully!")

if __name__ == "__main__":
    test_agent()