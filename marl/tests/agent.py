import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import build_env_from_config, build_policy_from_config, build_agent_from_config
from marl.agents.standard_agent import StandardAgent
from marl.algorithms.ppo import PPO

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def test_agent(cfg: DictConfig):
    """Test that an environment can be created from a config.yaml file using Hydra."""

    env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    env = build_env_from_config(env_config)

    policy_config = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = build_policy_from_config(policy_config)

    algorithm_config = OmegaConf.to_container(cfg.algorithm, resolve=True)
    algorithm = PPO()
    
    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    agent = StandardAgent(
        env,
        policy,
        algorithm,
        agent_config,
        algorithm_config
    )
    
    # Assert environment was created successfully
    assert env is not None
    
    env.close()
    print("Environment created successfully!")

if __name__ == "__main__":
    test_agent()