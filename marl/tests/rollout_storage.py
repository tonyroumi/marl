import torch
import pytest
import tempfile
import os
from unittest.mock import patch
import numpy as np

from marl.storage.rollout_storage import RolloutStorage, Transition
from marl.storage.world_model_storage import WorldModelStorage, WorldModelTransition

class TestWorldModelStorage:
    """Test suite for WorldModelStorage functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.capacity = 100
        self.state_dim = 4
        self.action_dim = 2
        self.buffer = WorldModelStorage(
            capacity=self.capacity,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
    
    def test_initialization(self):
        """Test buffer initialization."""
        assert self.buffer.capacity == self.capacity
        assert self.buffer.size == 0
        assert self.buffer.position == 0
        assert self.buffer.states.shape == (self.capacity, self.state_dim)
        assert self.buffer.actions.shape == (self.capacity, self.action_dim)
    
    def test_add_batch(self):
        """Test adding batches to buffer."""
        batch_size = 10
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.state_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.buffer.add_batch(states, actions, next_states, rewards, dones)
        
        assert self.buffer.size == batch_size
        assert self.buffer.position == batch_size
        
        # Check data integrity
        torch.testing.assert_close(self.buffer.states[:batch_size], states)
        torch.testing.assert_close(self.buffer.actions[:batch_size], actions)
        torch.testing.assert_close(self.buffer.next_states[:batch_size], next_states)
        torch.testing.assert_close(self.buffer.rewards[:batch_size], rewards)
        torch.testing.assert_close(self.buffer.dones[:batch_size], dones)
    
    def test_circular_buffer_overflow(self):
        """Test circular buffer behavior when exceeding capacity."""
        # Fill buffer to capacity + 20
        total_samples = self.capacity + 20
        states = torch.randn(total_samples, self.state_dim)
        actions = torch.randn(total_samples, self.action_dim)
        next_states = torch.randn(total_samples, self.state_dim)
        rewards = torch.randn(total_samples, 1)
        dones = torch.randint(0, 2, (total_samples, 1)).float()
        
        self.buffer.add_batch(states, actions, next_states, rewards, dones)
        
        # Size should be capped at capacity
        assert self.buffer.size == self.capacity
        assert self.buffer.position == 20  # Wrapped around
        
        # Check that oldest data was overwritten
        # The first 20 entries should contain the newest data
        torch.testing.assert_close(
            self.buffer.states[:20], 
            states[self.capacity:self.capacity + 20]
        )
    
    def test_sample(self):
        """Test sampling from buffer."""
        batch_size = 20
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.state_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.buffer.add_batch(states, actions, next_states, rewards, dones)
        
        # Sample smaller batch
        sample_size = 10
        sample = self.buffer.sample(sample_size)
        
        assert isinstance(sample, WorldModelTransition)
        assert sample.states.shape == (sample_size, self.state_dim)
        assert sample.actions.shape == (sample_size, self.action_dim)
        assert sample.next_states.shape == (sample_size, self.state_dim)
        assert sample.rewards.shape == (sample_size, 1)
        assert sample.dones.shape == (sample_size, 1)
    
    def test_save_load(self):
        """Test saving and loading buffer."""
        # Add some data
        batch_size = 50
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.state_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.buffer.add_batch(states, actions, next_states, rewards, dones)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            filepath = tmp.name
        
        try:
            self.buffer.save(filepath)
            
            # Create new buffer and load
            new_buffer = WorldModelStorage(
                capacity=self.capacity,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
            new_buffer.load(filepath)
            
            # Check that data matches
            assert new_buffer.size == self.buffer.size
            assert new_buffer.position == self.buffer.position
            torch.testing.assert_close(
                new_buffer.states[:new_buffer.size], 
                self.buffer.states[:self.buffer.size]
            )
            torch.testing.assert_close(
                new_buffer.actions[:new_buffer.size], 
                self.buffer.actions[:self.buffer.size]
            )
            
        finally:
            os.unlink(filepath)


class TestRolloutStorageWorldModel:
    """Test suite for RolloutStorage with world model functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.num_envs = 4
        self.num_transitions_per_env = 8
        self.actor_obs_dim = 6
        self.critic_obs_dim = 6
        self.action_dim = 3
        
        self.storage = RolloutStorage(
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_transitions_per_env,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            enable_world_model=True,
            world_model_buffer_size=1000
        )
    
    def create_dummy_transition(self):
        """Create a dummy transition for testing."""
        return Transition(
            actor_observations=torch.randn(self.num_envs, self.actor_obs_dim),
            critic_observations=torch.randn(self.num_envs, self.critic_obs_dim),
            actions=torch.randn(self.num_envs, self.action_dim),
            rewards=torch.randn(self.num_envs, 1),
            dones=torch.randint(0, 2, (self.num_envs, 1)).float(),
            values=torch.randn(self.num_envs, 1),
            actions_log_prob=torch.randn(self.num_envs, 1),
            action_mean=torch.randn(self.num_envs, self.action_dim),
            action_sigma=torch.randn(self.num_envs, self.action_dim),
        )
    
    def test_initialization_with_world_model(self):
        """Test initialization with world model enabled."""
        assert self.storage.enable_world_model == True
        assert hasattr(self.storage, 'world_model_buffer')
        assert hasattr(self.storage, 'next_observations')
        assert self.storage.next_observations.shape == (
            self.num_transitions_per_env, self.num_envs, self.actor_obs_dim
        )
    
    def test_initialization_without_world_model(self):
        """Test initialization with world model disabled."""
        storage = RolloutStorage(
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_transitions_per_env,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            enable_world_model=False
        )
        
        assert storage.enable_world_model == False
        assert not hasattr(storage, 'world_model_buffer')
        assert not hasattr(storage, 'next_observations')
    
    def test_add_next_observations(self):
        """Test adding next observations."""
        # Add a transition first
        transition = self.create_dummy_transition()
        self.storage.add(transition)
        
        # Add next observations
        next_obs = torch.randn(self.num_envs, self.actor_obs_dim)
        self.storage.add_next_observations(next_obs)
        
        torch.testing.assert_close(
            self.storage.next_observations[0], next_obs
        )
    
    def test_transfer_to_world_model_buffer(self):
        """Test transferring rollout data to world model buffer."""
        # Fill storage with some data
        for i in range(4):  # Add 4 transitions
            transition = self.create_dummy_transition()
            self.storage.add(transition)
            
            # Add next obs for all but the last transition
            if i < 3:
                next_obs = torch.randn(self.num_envs, self.actor_obs_dim)
                self.storage.add_next_observations(next_obs)
        
        initial_buffer_size = self.storage.world_model_buffer.size
        self.storage.transfer_to_world_model_buffer()
        
        # Should have added 3 complete transitions (3 steps * 4 envs = 12 samples)
        # because we only have next_observations for the first 3 transitions
        expected_added = 4 * self.num_envs
        assert self.storage.world_model_buffer.size == initial_buffer_size + expected_added
    
    def test_world_model_batch_generator(self):
        """Test world model batch generation."""
        # Fill world model buffer with some data
        batch_size = 10
        states = torch.randn(batch_size, self.actor_obs_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.actor_obs_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.storage.world_model_buffer.add_batch(
            states, actions, next_states, rewards, dones
        )
        
        # Generate batch
        sample_batch = self.storage.world_model_batch_generator(5)
        
        assert isinstance(sample_batch, WorldModelTransition)
        assert sample_batch.states.shape == (5, self.actor_obs_dim)
        assert sample_batch.actions.shape == (5, self.action_dim)
    
    def test_world_model_batch_generator_disabled(self):
        """Test world model batch generation when disabled."""
        storage = RolloutStorage(
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_transitions_per_env,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            enable_world_model=False
        )
        
        with pytest.raises(ValueError, match="World model is not enabled"):
            storage.world_model_batch_generator(10)
    
    def test_save_load_world_model_data(self):
        """Test saving and loading world model data."""
        # Add some data to world model buffer
        batch_size = 20
        states = torch.randn(batch_size, self.actor_obs_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.actor_obs_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.storage.world_model_buffer.add_batch(
            states, actions, next_states, rewards, dones
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            filepath = tmp.name
        
        try:
            self.storage.save_world_model_data(filepath)
            
            # Create new storage and load
            new_storage = RolloutStorage(
                num_envs=self.num_envs,
                num_transitions_per_env=self.num_transitions_per_env,
                actor_obs_dim=self.actor_obs_dim,
                critic_obs_dim=self.critic_obs_dim,
                action_dim=self.action_dim,
                device=self.device,
                enable_world_model=True,
                world_model_buffer_size=1000
            )
            new_storage.load_world_model_data(filepath)
            
            # Check that data matches
            assert new_storage.world_model_buffer.size == self.storage.world_model_buffer.size
            
        finally:
            os.unlink(filepath)
    
    def test_get_world_model_statistics(self):
        """Test world model statistics."""
        # Add some data
        batch_size = 30
        states = torch.randn(batch_size, self.actor_obs_dim)
        actions = torch.randn(batch_size, self.action_dim)
        next_states = torch.randn(batch_size, self.actor_obs_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        
        self.storage.world_model_buffer.add_batch(
            states, actions, next_states, rewards, dones
        )
        
        stats = self.storage.get_world_model_statistics()
        
        assert 'size' in stats
        assert 'capacity' in stats
        assert 'utilization' in stats
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert 'done_rate' in stats
        
        assert stats['size'] == batch_size
        assert stats['capacity'] == 1000
        assert stats['utilization'] == batch_size / 1000
    
    def test_backward_compatibility(self):
        """Test that original functionality still works."""
        # This should work exactly as before
        for i in range(5):
            transition = self.create_dummy_transition()
            self.storage.add(transition)
        
        # Compute returns (original functionality)
        last_values = torch.randn(self.num_envs, 1)
        self.storage.compute_returns(
            last_values=last_values,
            gamma=0.99,
            lambda_=0.95
        )
        
        # Generate mini-batches (original functionality)
        batches = list(self.storage.mini_batch_generator(num_mini_batches=2, num_epochs=1))
        assert len(batches) == 2
        
        # Check batch structure
        batch = batches[0]
        assert len(batch) == 9  # Expected number of elements
        
        # Clear and verify
        self.storage.clear()
        assert self.storage.step == 0


def test_integration_workflow():
    """Test complete integration workflow."""
    device = torch.device('cpu')
    storage = RolloutStorage(
        num_envs=2,
        num_transitions_per_env=4,
        actor_obs_dim=3,
        critic_obs_dim=3,
        action_dim=2,
        device=device,
        enable_world_model=True,
        world_model_buffer_size=100
    )
    
    # Simulate rollout collection
    for step in range(4):
        # Create transition
        transition = Transition(
            actor_observations=torch.randn(2, 3),
            critic_observations=torch.randn(2, 3),
            actions=torch.randn(2, 2),
            rewards=torch.randn(2, 1),
            dones=torch.randint(0, 2, (2, 1)).float(),
            values=torch.randn(2, 1),
            actions_log_prob=torch.randn(2, 1),
            action_mean=torch.randn(2, 2),
            action_sigma=torch.randn(2, 2)
        )
        
        storage.add(transition)
        
        # Add next observations for all but the last step
        if step < 3:
            next_obs = torch.randn(2, 3)
            storage.add_next_observations(next_obs)
    
    # Transfer to world model buffer
    initial_size = storage.get_world_model_buffer_size()
    storage.transfer_to_world_model_buffer()
    final_size = storage.get_world_model_buffer_size()
    
    # Should have added 3 complete transitions * 2 envs = 6 samples
    # (steps 0, 1, 2 have next_observations, step 3 doesn't)
    assert final_size == initial_size + 8
    
    # Test world model sampling
    batch = storage.world_model_batch_generator(4)
    assert batch.states.shape == (4, 3)
    assert batch.actions.shape == (4, 2)
    
    # Test statistics
    stats = storage.get_world_model_statistics()
    assert stats['size'] == 8
    assert 'mean_reward' in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])