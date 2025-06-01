import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import mujoco
from mujoco import _functions as mjf

from marl.agents.base_marl import BaseMARLAgent
from marl.policies.base_policy import BasePolicy
from marl.algorithms.base import BaseAlgorithm
from marl.storage.VAE_rBuffer import VAEBuf
from marl.networks.CLAS_vae import CLASVAE  # Assuming CLASVAE is defined in this module

def _find_raw_robosuite_env(wrapper_env):
    """
    Unwrap Gym wrappers (EpisodeStatsWrapper, GymWrapper, etc.) until you find
    the Robosuite environment instance that has .sim. Return that instance.
    """
    current = wrapper_env
    while hasattr(current, "env") and not hasattr(current, "sim"):
        current = current.env
    if not hasattr(current, "sim"):
        raise RuntimeError(f"Could not find .sim inside {wrapper_env!r}")
    return current

def world_to_local(p_world: np.ndarray,
                   p_body:  np.ndarray,
                   q_body:  np.ndarray) -> np.ndarray:
    """
    Convert a point p_world (in world coords) into the local frame of a body
    whose world‐frame origin is p_body and whose orientation quaternion is q_body.
    Returns a (3,) array in the body’s local coordinates.
    """
    Δ = p_world - p_body   # 3‐vector from body origin to world point
    w, x, y, z = q_body
    # Inverse (conjugate) of the unit quaternion q_body
    q_inv = np.array([w, -x, -y, -z], dtype=np.float64)

    # Helper to perform Hamilton product of two quaternions [w,x,y,z]
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z], dtype=np.float64)

    # Represent Δ as a “pure quaternion” [0, Δ_x, Δ_y, Δ_z]
    p_quat = np.concatenate([[0.0], Δ])

    # Rotate Δ by q_inv on the left and q_body on the right:
    tmp           = quat_mul(q_inv, p_quat)
    p_local_quat  = quat_mul(tmp, q_body)

    # p_local_quat = [0, x', y', z'], so return (x', y', z') as a (3,) array
    return p_local_quat[1:4]

class CLASVAEAgent(BaseMARLAgent):
    """
    CLAS VAE Agent that extends BaseMARLAgent for VAE-based multi-agent learning.
    
    This agent:
    1. Collects experience in a VAE-specific replay buffer
    2. Trains a CLAS VAE model to learn latent action representations
    3. Uses the trained VAE for coordinated multi-agent control
    """

    def __init__(
        self,
        env: Any,
        policy: BasePolicy,
        observation_config: Dict[str, List[str]],
        num_transitions_per_env: int,
        normalize_observations: bool,
        vae_config: Dict[str, Any],
        vae_buffer_size: int = 100000,
        vae_batch_size: int = 64,
        vae_train_freq: int = 1,
        min_buffer_size: int = 1000,
        algorithm: Optional[BaseAlgorithm] = None,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            vae_config: Configuration dictionary for CLASVAE
            vae_buffer_size: Size of VAE replay buffer
            vae_batch_size: Batch size for VAE training
            vae_train_freq: Frequency of VAE training (every N environment steps)
            min_buffer_size: Minimum buffer size before starting VAE training
        """
        # Initialize base class
        super().__init__(
            env=env,
            policy=policy,
            algorithm=algorithm,
            observation_config=observation_config,
            num_transitions_per_env=num_transitions_per_env,
            normalize_observations=normalize_observations,
            logger=logger,
            device=device
        )
        
        # VAE-specific initialization
        self.vae_config = vae_config
        self.vae_batch_size = vae_batch_size
        self.vae_train_freq = vae_train_freq
        self.min_buffer_size = min_buffer_size
        
        # Initialize VAE buffer
        self.vae_buffer = VAEBuf(capacity=vae_buffer_size)
        
        # Initialize CLAS VAE
        self.vae = CLASVAE(config=vae_config)
        
        # Training counters
        self.step_count = 0
        self.vae_training_steps = 0
        
        # Training metrics
        self.vae_losses = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }

    def store_transition(self, obs_dict: Dict[str, np.ndarray], actions: np.ndarray):
        """
        Store a transition in the VAE buffer.
        
        Args:
            obs_dict: Raw observation dictionary from environment
            actions: Combined actions from both robots [robot0_action, robot1_action]
        """
        # Ensure actions are in the right format (numpy array)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # Store in VAE buffer
        self.vae_buffer.push(obs_dict, actions)
        
        
    def reset_and_weld(self, pd_steps: int = 20, kp: float = 50.0):
        """
        1) Unwrap SyncVectorEnv → raw robsuite env, reset (randomizes pot & handles).
        2) Run pd_steps of Cartesian PD so each EEF site overlaps its pot-handle site
           in world-space (distance = 0, position-only).
        3) Grab the in-memory MJCF via raw_env.sim.model.get_xml(), then inject two
           <connect site1="…" site2="…"/> tags under <equality> (or create one if none exists).
        4) Recompile via MjModel.from_xml_string(modified_xml), allocate MjData, then
           build a fresh simulator using the **same class** as raw_env.sim.
        5) **Swap** that new simulator into raw_env (not self.env), step once, and return obs.
        """

        # ------------------------------------------------------
        # 1) UNWRAP & RESET
        # ------------------------------------------------------
        # self.env is a SyncVectorEnv of GymWrappers → raw_env
        wrapped = self.env.envs[0]
        raw_env = _find_raw_robosuite_env(wrapped)

        obs_dict, _ = raw_env.reset()  # randomize pot+handles
        model = raw_env.sim.model
        data  = raw_env.sim.data

        # ------------------------------------------------------
        # 2) RUN PD so EEF site = handle site
        # ------------------------------------------------------
        eef0_id     = model.site_name2id("robot0_right_center")
        eef1_id     = model.site_name2id("robot1_right_center")
        handle0_id  = model.site_name2id("pot_handle0")
        handle1_id  = model.site_name2id("pot_handle1")

        nu           = model.nu
        acts_per_arm = nu // 2  # e.g. 9 actuators per Sawyer arm

        for _ in range(pd_steps):
            obs = raw_env._get_observations()

            # — Arm 0: robot0_right_center → pot_handle0 —
            target0 = obs["handle0_xpos"].reshape(-1)
            curr0   = obs["robot0_eef_pos"].reshape(-1)
            err0    = target0 - curr0  # (3,)

            J0_full = raw_env.sim.data.get_body_jacp("robot0_right_hand")
            J0_pos  = J0_full[:, 0:7]                              # (3,7)
            dq0     = kp * (J0_pos.T.dot(err0))                    # (7,)

            # — Arm 1: robot1_right_center → pot_handle1 —
            target1 = obs["handle1_xpos"].reshape(-1)
            curr1   = obs["robot1_eef_pos"].reshape(-1)
            err1    = target1 - curr1  # (3,)

            J1_full = raw_env.sim.data.get_body_jacp("robot1_right_hand")
            J1_pos  = J1_full[:, acts_per_arm : acts_per_arm + 7]  # (3,7)
            dq1     = kp * (J1_pos.T.dot(err1))                    # (7,)

            ctrl = np.zeros(nu, dtype=np.float64)
            ctrl[0:7]                           = dq0
            ctrl[acts_per_arm : acts_per_arm + 7] = dq1
            raw_env.sim.data.ctrl[:] = ctrl
            raw_env.sim.step()

        # Now data.site_xpos[eef0_id] == data.site_xpos[handle0_id], etc.

        # ------------------------------------------------------
        # 3) READ MJCF XML & INJECT two <connect site1=… site2=…/>
        # ------------------------------------------------------
        xml_string = raw_env.sim.model.get_xml()
        if "<equality>" in xml_string:
            head, tail = xml_string.split("</equality>", 1)
            weld_lines = """
  <connect site1="robot0_right_center" site2="pot_handle0" active="true"/>
  <connect site1="robot1_right_center" site2="pot_handle1" active="true"/>
"""
            modified_xml = head + weld_lines + "</equality>" + tail
        else:
            part1, part2 = xml_string.split("</worldbody>", 1)
            prefix = part1 + "</worldbody>\n"
            suffix = part2
            weld_block = """
  <equality>
    <connect site1="robot0_right_center" site2="pot_handle0" active="true"/>
    <connect site1="robot1_right_center" site2="pot_handle1" active="true"/>
  </equality>
"""
            modified_xml = prefix + weld_block + suffix

        # ------------------------------------------------------
        # 4) RECOMPILE into a new MjModel + MjData
        # ------------------------------------------------------
        new_model = mujoco.MjModel.from_xml_string(modified_xml)
        new_data  = mujoco.MjData(new_model)

        # ------------------------------------------------------
        # 5) INSTANTIATE a brand‐new simulator using the same class as raw_env.sim
        # ------------------------------------------------------
        SimClass = raw_env.sim.__class__          # grab whatever class raw_env.sim actually is
        new_sim  = SimClass(new_model)  # e.g. mujoco.MjSim or a wrapped variant

        # >>>>> **THIS IS THE CRITICAL CHANGE** <<<<<
        # Swap the simulator into the raw Robosuite env, not the outer wrapper:
        raw_env.sim   = new_sim
        raw_env.model = new_sim.model
        if hasattr(raw_env, "mujoco_model"):
            raw_env.mujoco_model = new_model

        # ------------------------------------------------------
        # 6) STEP ONCE (to prime MuJoCo’s constraint solver), then return obs
        # ------------------------------------------------------
        raw_env.sim.step()
        final_obs = raw_env._get_observations()
        return final_obs

        
    def _prefill_buffer(self,
                        prefill_size: int = 50_000,
                        max_episode_steps: int = 200) -> None:
        """
        Step the env with a hand-coded / random policy until
        self.vae_buffer contains `prefill_size` transitions.
        """
        if self.logger:
            self.logger.info(f"Prefilling VAE buffer with {prefill_size} transitions")

        
        obs_dict = self.env.reset()  # Reset and weld the robots to the handle
        steps_in_ep = 0

        pbar = tqdm(total=prefill_size, desc="Prefill", unit="step")
        
        while len(self.vae_buffer) < prefill_size:
            # ----- cheap exploration policy -----
            # completely random joint velocities in [-1, 1]
            rand_u = np.random.uniform(-1.0, 1.0, size=(1, 14))
            # rand_u = np.zeros((1, 14))
            # OR: a scripted grab-and-weld controller you already have
            # rand_u = my_scripted_weld_controller(obs_dict)

            next_obs, reward, done, truncated, info = self.env.step(rand_u)
            self.store_transition(obs_dict, rand_u)
            pbar.update(1)

            obs_dict = next_obs
            steps_in_ep += 1

            if done or steps_in_ep >= max_episode_steps:
                obs_dict, infos = self.env.reset()
                steps_in_ep = 0

        pbar.close()
        if self.logger:
            self.logger.info("Prefill done ✓")

    def train_vae_step(self) -> Dict[str, float]:
        """
        Perform one VAE training step if buffer has enough samples.
        
        Returns:
            Dictionary of training losses, or empty dict if training was skipped
        """
        if not self.vae_buffer.is_ready(self.min_buffer_size):
            if self.logger:
                self.logger.info("VAE buffer not ready for training, skipping step")
            return {}
        
        obs_dicts, actions = self.vae_buffer.sample_batch(self.vae_batch_size)
            
        # Convert list of obs_dicts to batched format
        batched_obs_dict = self._batch_observations(obs_dicts)
        
        # Train VAE
        losses = self.vae.train_step(batched_obs_dict, actions)
        
        # Update training counter
        self.vae_training_steps += 1
        
        # Store losses for logging
        for key, value in losses.items():
            self.vae_losses[key].append(value)
        
        try:
            # Sample batch from buffer
            obs_dicts, actions = self.vae_buffer.sample_batch(self.vae_batch_size)
            
            # Convert list of obs_dicts to batched format
            batched_obs_dict = self._batch_observations(obs_dicts)
            
            # Train VAE
            losses = self.vae.train_step(batched_obs_dict, actions)
            
            # Update training counter
            self.vae_training_steps += 1
            
            # Store losses for logging
            for key, value in losses.items():
                self.vae_losses[key].append(value)
        
            return losses
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"VAE training step failed: {str(e)}")
            return {}

    def _batch_observations(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Convert list of observation dictionaries to batched format.
        
        Args:
            obs_list: List of observation dictionaries
            
        Returns:
            Dictionary with batched observations
        """
        if not obs_list:
            return {}
        
        batched_obs = {}
        for key in obs_list[0].keys():
            # Stack observations across batch dimension
            arr = np.stack([obs[key] for obs in obs_list], axis=0)
            
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]    # now (B, D)

            batched_obs[key] = arr
        
        return batched_obs

    def get_latent_actions(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Get latent actions from the VAE prior for coordination.
        
        Args:
            obs_dict: Current observation dictionary
            
        Returns:
            Sampled latent actions for coordination
        """
        # Parse observations
        _, _, _, full_obs = self.vae.parse_observation(obs_dict)
        
        with torch.no_grad():
            # Sample from prior p(v|o)
            mu_prior, logvar_prior = self.vae.prior(full_obs)
            latent_pre_tanh = self.vae.reparameterize(mu_prior, logvar_prior)
            latent_action = torch.tanh(latent_pre_tanh)
        
        return latent_action

    def decode_coordinated_actions(
        self, 
        obs_dict: Dict[str, np.ndarray], 
        latent_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent actions to robot-specific actions using trained VAE.
        
        Args:
            obs_dict: Current observations
            latent_action: Latent coordination signal
            
        Returns:
            Tuple of (robot0_actions, robot1_actions)
        """
        return self.vae.decode_actions(obs_dict, latent_action)

    def learn(self,
          prefill_size: int     = 1000,
          vae_updates:   int     = 10000,
          max_episode_steps: int = 200) -> None:
        """
        1) Collect `prefill_size` transitions with a random / scripted controller
        2) Run exactly `vae_updates` gradient steps on the VAE
        (sampling fresh batches from the filled buffer)
        """

        # ------------- Phase 1: prefilling ----------
        self._prefill_buffer(prefill_size, max_episode_steps)

        print("now training the VAE with {} updates".format(vae_updates))
        # ------------- Phase 2: VAE updates only ----------
        for update in range(vae_updates):
            losses = self.train_vae_step()           # draws a batch each call
            if self.logger and update % 500 == 0:
                self.logger.info(
                    f"VAE-update {update}/{vae_updates} "
                    f"| total: {losses.get('total_loss', 0):.4f} "
                    f"| recon: {losses.get('recon_loss', 0):.4f} "
                    f"| KL:    {losses.get('kl_loss', 0):.4f}"
                )

        # ------------- save weights for later RL ----------
        self.save_vae("clas_vae_prefilled.pt")



    ################################################
    ############### Utility Methods
    ################################################



    def save_vae(self, path: str):
        """Save the trained VAE model"""
        torch.save({
            'encoder': self.vae.encoder.state_dict(),
            'decoder0': self.vae.decoder0.state_dict(),
            'decoder1': self.vae.decoder1.state_dict(),
            'prior': self.vae.prior.state_dict(),
            'vae_config': self.vae_config,
            'training_steps': self.vae_training_steps
        }, path)
        
        if self.logger:
            self.logger.info(f"VAE model saved to {path}")

    def load_vae(self, path: str):
        """Load a pre-trained VAE model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.vae.encoder.load_state_dict(checkpoint['encoder'])
        self.vae.decoder0.load_state_dict(checkpoint['decoder0'])
        self.vae.decoder1.load_state_dict(checkpoint['decoder1'])
        self.vae.prior.load_state_dict(checkpoint['prior'])
        self.vae_training_steps = checkpoint.get('training_steps', 0)
        
        if self.logger:
            self.logger.info(f"VAE model loaded from {path}")

    def get_vae_metrics(self) -> Dict[str, float]:
        """Get recent VAE training metrics"""
        if not any(self.vae_losses.values()):
            return {}
        
        return {
            f'vae/{key}_recent': np.mean(values[-100:]) if values else 0.0
            for key, values in self.vae_losses.items()
        }

    def eval_mode(self):
        """Set agent to evaluation mode"""
        super().eval_mode()
        # Set VAE to eval mode
        self.vae.encoder.eval()
        self.vae.decoder0.eval()
        self.vae.decoder1.eval()
        self.vae.prior.eval()

    def train_mode(self):
        """Set agent to training mode"""
        super().train_mode()
        # Set VAE to train mode
        self.vae.encoder.train()
        self.vae.decoder0.train()
        self.vae.decoder1.train()
        self.vae.prior.train()