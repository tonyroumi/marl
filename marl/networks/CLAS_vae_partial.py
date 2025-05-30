# NOTE THIS IS NOT THE FINAL VERSION, IT IS A WORK IN PROGRESS


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

"""
INPUT: full observation (joint angles, EE pose, object pose, obj_handle pose) (concatenated from all agents), 
        actions (as joint velocities) (concatenated from all agents)
OUTPUT: latent action space parameters (mu, logvar) (non-semantic meaning here is the wrench on the object)
"""
class CLASEncoder(nn.Module):
    """Encoder network that maps observations and actions to latent space"""
    
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Input: concatenated observations and actions from all agents
        input_dim = obs_dim + action_dim
        
        # 3 hidden layers as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log variance for latent distribution
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Full observation (concatenated from all agents)
            actions: Actions from all agents
        Returns:
            mu, logvar: Parameters of latent distribution (before tanh)
        """
        x = torch.cat([obs, actions], dim=-1)
        hidden = self.layers(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

"""
INPUT: full observation o (joint angles, EE pose, object pose, obj_handle pose) (concatenated from all agents), 
        latent actions v_c (as joint velocities) (concatenated from all agents)
OUTPUT: original action
"""
class CLASDecoder(nn.Module):
    """Decoder network that maps latent actions to original action space"""
    
    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 256, learn_variance: bool = False):
        super().__init__()
        
        self.learn_variance = learn_variance
        self.action_dim = action_dim
        
        # Input: observation + latent action
        input_dim = obs_dim + latent_dim
        
        # 3 hidden layers as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        if self.learn_variance:
            self.logvar_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor, latent_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Observation for specific agent
            latent_action: Latent action from shared policy (already tanh-squashed)
        Returns:
            mu, logvar: Parameters of action distribution (before tanh)
        """
        x = torch.cat([obs, latent_action], dim=-1)
        hidden = self.layers(x)
        mu = self.mu_layer(hidden)
        
        if self.learn_variance:
            logvar = self.logvar_layer(hidden)
        else:
            # Fixed variance for simpler training
            logvar = torch.zeros_like(mu)
            
        return mu, logvar

class CLASPrior(nn.Module):
    """Prior network p(v|o) - policy-like form"""
    
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 2 hidden layers for policy networks as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu, logvar: Parameters of prior distribution (before tanh)
        """
        hidden = self.layers(obs)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

class CLASPolicy(nn.Module):
    """Shared policy that outputs latent actions"""
    
    def __init__(self, shared_obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 2 hidden layers for policy networks
        self.layers = nn.Sequential(
            nn.Linear(shared_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, shared_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Returns:
            latent_action: Sampled and tanh-squashed latent action
        """
        hidden = self.layers(shared_obs)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        if deterministic:
            # For evaluation, use mean
            pre_tanh = mu
        else:
            # Sample from Gaussian and apply tanh
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            pre_tanh = mu + eps * std
        
        # Apply tanh transformation
        latent_action = torch.tanh(pre_tanh)
        return latent_action

class CLASVAE:
    """Complete CLAS VAE system for two-arm manipulation"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dictionary containing network configurations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract dimensions from config
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config.get('hidden_dim', 256)
        self.learn_decoder_variance = config.get('learn_decoder_variance', False)
        
        # Calculate observation dimensions from your data structure
        self.robot0_obs_dim = self._calculate_robot_obs_dim()
        self.robot1_obs_dim = self._calculate_robot_obs_dim()
        self.shared_obs_dim = self._calculate_shared_obs_dim()
        self.full_obs_dim = self.robot0_obs_dim + self.robot1_obs_dim + self.shared_obs_dim
        
        # Action dimensions (assuming 7 DOF for each robot arm)
        self.robot0_action_dim = 7
        self.robot1_action_dim = 7
        self.total_action_dim = self.robot0_action_dim + self.robot1_action_dim
        
        # Initialize networks
        self.encoder = CLASEncoder(
            obs_dim=self.full_obs_dim,
            action_dim=self.total_action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.decoder0 = CLASDecoder(
            obs_dim=self.robot0_obs_dim + self.shared_obs_dim,
            latent_dim=self.latent_dim,
            action_dim=self.robot0_action_dim,
            hidden_dim=self.hidden_dim,
            learn_variance=self.learn_decoder_variance
        ).to(self.device)
        
        self.decoder1 = CLASDecoder(
            obs_dim=self.robot1_obs_dim + self.shared_obs_dim,
            latent_dim=self.latent_dim,
            action_dim=self.robot1_action_dim,
            hidden_dim=self.hidden_dim,
            learn_variance=self.learn_decoder_variance
        ).to(self.device)
        
        self.prior = CLASPrior(
            obs_dim=self.full_obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.policy = CLASPolicy(
            shared_obs_dim=self.shared_obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.get('lr', 3e-4))
        self.decoder_optimizer = optim.Adam(
            list(self.decoder0.parameters()) + list(self.decoder1.parameters()), 
            lr=config.get('lr', 3e-4)
        )
        self.prior_optimizer = optim.Adam(self.prior.parameters(), lr=config.get('lr', 3e-4))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.get('lr', 3e-4))
        
    def _calculate_robot_obs_dim(self) -> int:
        """Calculate observation dimension for one robot based on your data structure"""
        # From your data: joint_pos(7) + joint_pos_cos(7) + joint_pos_sin(7) + joint_vel(7) + 
        # eef_pos(3) + eef_quat(4) + eef_quat_site(4) + gripper_qpos(2) + gripper_qvel(2)
        return 7 + 7 + 7 + 7 + 3 + 4 + 4 + 2 + 2  # Total: 43
    
    def _calculate_shared_obs_dim(self) -> int:
        """Calculate shared observation dimension"""
        # From your data: pot_pos(3) + pot_quat(4) + handle positions and gripper relations
        # pot_pos(3) + pot_quat(4) + handle0_xpos(3) + handle1_xpos(3) + 
        # gripper0_to_handle0(3) + gripper1_to_handle1(3)
        return 3 + 4 + 3 + 3 + 3 + 3  # Total: 19
    
    def parse_observation(self, obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse observation dictionary into robot-specific and shared observations
        
        Returns:
            robot0_obs, robot1_obs, shared_obs, full_obs
        """
        # Robot 0 observations
        robot0_obs = torch.cat([
            torch.tensor(obs_dict['robot0_joint_pos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_joint_pos_cos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_joint_pos_sin'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_joint_vel'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_eef_pos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_eef_quat'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_eef_quat_site'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_gripper_qpos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot0_gripper_qvel'], dtype=torch.float32),
        ], dim=-1)
        
        # Robot 1 observations
        robot1_obs = torch.cat([
            torch.tensor(obs_dict['robot1_joint_pos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_joint_pos_cos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_joint_pos_sin'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_joint_vel'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_eef_pos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_eef_quat'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_eef_quat_site'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_gripper_qpos'], dtype=torch.float32),
            torch.tensor(obs_dict['robot1_gripper_qvel'], dtype=torch.float32),
        ], dim=-1)
        
        # Shared observations (task-related)
        shared_obs = torch.cat([
            torch.tensor(obs_dict['pot_pos'], dtype=torch.float32),
            torch.tensor(obs_dict['pot_quat'], dtype=torch.float32),
            torch.tensor(obs_dict['handle0_xpos'], dtype=torch.float32),
            torch.tensor(obs_dict['handle1_xpos'], dtype=torch.float32),
            torch.tensor(obs_dict['gripper0_to_handle0'], dtype=torch.float32),
            torch.tensor(obs_dict['gripper1_to_handle1'], dtype=torch.float32),
        ], dim=-1)
        
        # Full observation (concatenated)
        full_obs = torch.cat([robot0_obs, robot1_obs, shared_obs], dim=-1)
        
        return robot0_obs, robot1_obs, shared_obs, full_obs
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def tanh_gaussian_log_prob(self, pre_tanh: torch.Tensor, tanh_action: torch.Tensor, 
                               mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of tanh-transformed Gaussian
        
        Args:
            pre_tanh: Action before tanh transformation
            tanh_action: Action after tanh transformation
            mu, logvar: Gaussian parameters
        """
        # Standard Gaussian log prob
        log_prob = -0.5 * (((pre_tanh - mu) / torch.exp(0.5 * logvar)) ** 2 + 
                          logvar + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Correction for tanh transformation (log Jacobian)
        log_prob -= torch.sum(torch.log(1 - tanh_action ** 2 + 1e-6), dim=-1, keepdim=True)
        
        return log_prob
    
    def vae_loss(self, obs_dict: Dict, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss as described in the paper (Equation 2)
        
        Args:
            obs_dict: Observation dictionary
            actions: Actions taken by both robots [robot0_action, robot1_action] (already tanh-squashed)
        """
        robot0_obs, robot1_obs, shared_obs, full_obs = self.parse_observation(obs_dict)
        
        # Move to device
        robot0_obs = robot0_obs.to(self.device)
        robot1_obs = robot1_obs.to(self.device)
        shared_obs = shared_obs.to(self.device)
        full_obs = full_obs.to(self.device)
        actions = actions.to(self.device)
        
        # Encoder: q(v|o,u) - get latent distribution parameters
        mu_enc, logvar_enc = self.encoder(full_obs, actions)
        
        # Sample latent action (before tanh)
        latent_pre_tanh = self.reparameterize(mu_enc, logvar_enc)
        latent_action = torch.tanh(latent_pre_tanh)
        
        # Decoders: p(u|o,v) - get action distribution parameters
        mu_dec_0, logvar_dec_0 = self.decoder0(
            torch.cat([robot0_obs, shared_obs], dim=-1), 
            latent_action
        )
        mu_dec_1, logvar_dec_1 = self.decoder1(
            torch.cat([robot1_obs, shared_obs], dim=-1), 
            latent_action
        )
        
        # Prior: p(v|o)
        mu_prior, logvar_prior = self.prior(full_obs)
        
        # Split actions
        robot0_action = actions[:, :self.robot0_action_dim]
        robot1_action = actions[:, self.robot0_action_dim:]
        
        # Reconstruction loss
        if self.learn_decoder_variance:
            # Full Gaussian likelihood with learned variance
            # This would require inverse tanh to get pre_tanh actions
            # For simplicity, using MSE here but you could implement full likelihood
            recon_loss_0 = F.mse_loss(torch.tanh(mu_dec_0), robot0_action)
            recon_loss_1 = F.mse_loss(torch.tanh(mu_dec_1), robot1_action)
        else:
            # Fixed variance - use MSE loss (equivalent to unit variance Gaussian)
            recon_loss_0 = F.mse_loss(torch.tanh(mu_dec_0), robot0_action)
            recon_loss_1 = F.mse_loss(torch.tanh(mu_dec_1), robot1_action)
        
        recon_loss = recon_loss_0 + recon_loss_1
        
        # KL divergence between encoder and prior (both before tanh)
        # KL(q(v|o,u) || p(v|o))
        kl_loss = -0.5 * torch.sum(
            1 + logvar_enc - logvar_prior - 
            ((mu_enc - mu_prior).pow(2) + logvar_enc.exp()) / logvar_prior.exp(),
            dim=-1
        ).mean()
        
        total_loss = recon_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'recon_loss_0': recon_loss_0,
            'recon_loss_1': recon_loss_1
        }
    
    def train_step(self, obs_dict: Dict, actions: torch.Tensor) -> Dict[str, float]:
        """Single training step for the VAE"""
        
        # Compute losses
        losses = self.vae_loss(obs_dict, actions)
        
        # Update all networks together (as they're coupled)
        all_params = (list(self.encoder.parameters()) + 
                     list(self.decoder0.parameters()) + 
                     list(self.decoder1.parameters()) + 
                     list(self.prior.parameters()))
        
        all_optimizer = optim.Adam(all_params, lr=3e-4)
        
        all_optimizer.zero_grad()
        losses['total_loss'].backward()
        all_optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}
    
    def get_latent_action(self, shared_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get latent action from shared policy"""
        shared_obs = shared_obs.to(self.device)
        with torch.no_grad():
            latent_action = self.policy(shared_obs, deterministic=deterministic)
        return latent_action
    
    def decode_actions(self, obs_dict: Dict, latent_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent action to robot-specific actions"""
        robot0_obs, robot1_obs, shared_obs, _ = self.parse_observation(obs_dict)
        
        robot0_obs = robot0_obs.to(self.device)
        robot1_obs = robot1_obs.to(self.device)
        shared_obs = shared_obs.to(self.device)
        latent_action = latent_action.to(self.device)
        
        with torch.no_grad():
            mu_0, _ = self.decoder0(torch.cat([robot0_obs, shared_obs], dim=-1), latent_action)
            mu_1, _ = self.decoder1(torch.cat([robot1_obs, shared_obs], dim=-1), latent_action)
            
            # Apply tanh to get final actions
            robot0_action = torch.tanh(mu_0)
            robot1_action = torch.tanh(mu_1)
        
        return robot0_action, robot1_action

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'latent_dim': 6,  # Dimensionality of latent action space (should be < 14 for compression)
        'hidden_dim': 256,  # Hidden layer size
        'lr': 3e-4,  # Learning rate
        'learn_decoder_variance': False,  # Set to True for full Gaussian likelihood
    }
    
    # Initialize CLAS VAE
    clas_vae = CLASVAE(config)
    
    # Example training step
    # You would get these from your environment step
    obs_dict = {
        'robot0_joint_pos': np.random.randn(1, 7),
        'robot0_joint_pos_cos': np.random.randn(1, 7),
        'robot0_joint_pos_sin': np.random.randn(1, 7),
        'robot0_joint_vel': np.random.randn(1, 7),
        'robot0_eef_pos': np.random.randn(1, 3),
        'robot0_eef_quat': np.random.randn(1, 4),
        'robot0_eef_quat_site': np.random.randn(1, 4),
        'robot0_gripper_qpos': np.random.randn(1, 2),
        'robot0_gripper_qvel': np.random.randn(1, 2),
        'robot1_joint_pos': np.random.randn(1, 7),
        'robot1_joint_pos_cos': np.random.randn(1, 7),
        'robot1_joint_pos_sin': np.random.randn(1, 7),
        'robot1_joint_vel': np.random.randn(1, 7),
        'robot1_eef_pos': np.random.randn(1, 3),
        'robot1_eef_quat': np.random.randn(1, 4),
        'robot1_eef_quat_site': np.random.randn(1, 4),
        'robot1_gripper_qpos': np.random.randn(1, 2),
        'robot1_gripper_qvel': np.random.randn(1, 2),
        'pot_pos': np.random.randn(1, 3),
        'pot_quat': np.random.randn(1, 4),
        'handle0_xpos': np.random.randn(1, 3),
        'handle1_xpos': np.random.randn(1, 3),
        'gripper0_to_handle0': np.random.randn(1, 3),
        'gripper1_to_handle1': np.random.randn(1, 3),
    }
    
    # Actions should be tanh-squashed (from your RL environment)
    actions = torch.tanh(torch.randn(1, 14))  # 7 for each robot
    
    # Training step
    losses = clas_vae.train_step(obs_dict, actions)
    print("Training losses:", losses)
    
    # Example inference
    _, _, shared_obs, _ = clas_vae.parse_observation(obs_dict)
    latent_action = clas_vae.get_latent_action(shared_obs, deterministic=True)
    robot0_action, robot1_action = clas_vae.decode_actions(obs_dict, latent_action)
    
    print(f"Latent action shape: {latent_action.shape}")
    print(f"Robot 0 action shape: {robot0_action.shape}")
    print(f"Robot 1 action shape: {robot1_action.shape}")