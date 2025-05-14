import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from marl.utils.config_utils import build_policy_from_config

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def verify_policy_connections(cfg: DictConfig):
    """Verify connections between components in a policy built with Hydra."""

    policy_cfg = OmegaConf.to_container(cfg.policies, resolve=True)
    
    policy = build_policy_from_config(policy_cfg)
    
    print(f"Successfully built policy from Hydra config")
    
    print(f"Policy components: {list(policy.components.keys())}")
    print(f"Policy connections: {policy.connections}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 10  # Adjust based on your actual config
    action_dim = 2  # Adjust based on your actual config
    batch_size = 5
    
    if "encoder" in policy.components and "agent" in policy.components:
        encoder_obs = torch.rand(batch_size, obs_dim).to(device)
        agent_obs = torch.rand(batch_size, obs_dim).to(device)
        
        obs = {
            "encoder": encoder_obs,
            "agent": agent_obs
        }
        
        # Get individual components
        encoder_component = policy.components["encoder"]
        agent_component = policy.components["agent"]
        
        print(f"\nTesting component interaction:")
        
        # Manually create the concatenated input for encoder
        agent_actions, _ = agent_component.act(obs["agent"], deterministic=True)
        print(f"Agent actions shape: {agent_actions.shape}")
        
        concatenated_input = torch.cat([agent_actions, obs['encoder']], dim=1)
        print(f"Concatenated input shape: {concatenated_input.shape}")
        
        # Test act method with deterministic flag
        encoder_out = encoder_component.forward(concatenated_input)
        print(f"Encoder output shape: {encoder_out.shape}")
        
        # Test entire policy forward
        policy_actions, _ = policy.act(obs, deterministic=True)
        print(f"Policy actions: {policy_actions.keys()}")
        print(f"Policy encoder output shape: {policy_actions['encoder'].shape}")
        
        # Verify outputs match
        outputs_match = torch.allclose(encoder_out, policy_actions["encoder"])
        print(f"Outputs match: {outputs_match}")
        
        # Test evaluate method
        agent_val = agent_component.evaluate(obs["agent"])
        policy_val = policy.evaluate(obs)
        
        print(f"Agent value shape: {agent_val.shape}")
        print(f"Policy agent value shape: {policy_val['agent'].shape}")
        values_match = torch.allclose(agent_val, policy_val["agent"])
        print(f"Values match: {values_match}")
    else:
        print(f"Warning: Expected 'encoder' and 'agent' components not found in policy")
        print(f"Available components: {list(policy.components.keys())}")

if __name__ == "__main__":
    verify_policy_connections()