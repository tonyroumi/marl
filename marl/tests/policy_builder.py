import unittest
import torch
from unittest.mock import  patch

from marl.policies import MultiAgentPolicyBuilder


class TestMultiAgentPolicyBuilder(unittest.TestCase):
    def setUp(self):
        # Initialize the builder before each test
        self.builder = MultiAgentPolicyBuilder()
        
        # Common test parameters
        self.obs_dim = 10
        self.action_dim = 4
        self.hidden_dims = [64, 32]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_add_single_actor_critic(self):
        """Test adding a simple actor-critic component."""
        policy = self.builder.add_component(
            component_id="agent1",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).build()
        
        # Check if component exists
        self.assertIn("agent1", self.builder.components)
        
        # Test forward method
        batch_size = 5
        obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        
        # Get the component network
        component = policy.components["agent1"]
        
        # Get outputs from direct component forward pass
        component_out, component_info = component.act(obs, deterministic=True)
        policy_out, policy_info = policy.act({"agent1": obs}, deterministic=True)

        # Verify policy outputs match direct component outputs
        torch.testing.assert_close(component_out, policy_out["agent1"])
        assert_dicts_close(component_info, policy_info["agent1"])

        # Test get_actions method with deterministic flag
        component_out = component.evaluate(obs)
        policy_out = policy.evaluate({"agent1": obs})

        torch.testing.assert_close(component_out, policy_out["agent1"])
        
        # Test get_actions method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        component_out, component_info = component.act(obs, deterministic=False)
        
        torch.manual_seed(42)
        policy_out, policy_info = policy.act({"agent1": obs}, deterministic=False)
        
        torch.testing.assert_close(component_out, policy_out["agent1"])
        assert_dicts_close(component_info, policy_info["agent1"])
        
        # Mock the save and load methods for testing
        with patch.object(policy, 'save', return_value=None) as mock_save:
            policy.save("test_path")
            mock_save.assert_called_once_with("test_path")
            
        with patch.object(policy, 'load', return_value=None) as mock_load:
            policy.load("test_path")
            mock_load.assert_called_once_with("test_path")

    def test_add_separate_actor_critic(self):
        """Test adding separate actor and critic components."""
        policy = self.builder.add_component(
            component_id="actor1",
            network_type="mlp",
            network_class="actor",
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            actor_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="critic1",
            network_type="mlp",
            network_class="critic",
            critic_obs_dim=self.obs_dim,
            critic_out_dim=1,  # Value function has output dim 1
            critic_hidden_dims=self.hidden_dims
        ).build()
        
        # Check if components exist
        self.assertIn("actor1", self.builder.components)
        self.assertIn("critic1", self.builder.components)
        
        # Test forward method
        batch_size = 5
        actor_obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        critic_obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        
        obs = {
            "actor1": actor_obs,
            "critic1": critic_obs
        }
        
        # Get individual components
        actor_component = policy.components["actor1"]
        critic_component = policy.components["critic1"]
        
        # Test act method with deterministic flag
        actor_out, actor_info = actor_component.act(actor_obs, deterministic=True)
        policy_out, policy_info = policy.act({"actor1": actor_obs}, deterministic=True)
        
        torch.testing.assert_close(actor_out, policy_out["actor1"])
        assert_dicts_close(actor_info, policy_info["actor1"])
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        actor_out_stoch, actor_info_stoch = actor_component.act(actor_obs, deterministic=False)
        
        torch.manual_seed(42)
        policy_out_stoch, policy_info_stoch = policy.act({"actor1": actor_obs}, deterministic=False)
        
        torch.testing.assert_close(actor_out_stoch, policy_out_stoch["actor1"])
        assert_dicts_close(actor_info_stoch, policy_info_stoch["actor1"])
        
        # Test evaluate method for critic
        critic_out = critic_component.evaluate(critic_obs)
        policy_out = policy.evaluate({"critic1": critic_obs})
        
        torch.testing.assert_close(critic_out, policy_out["critic1"])
        
        # Mock the save and load methods for testing
        with patch.object(policy, 'save', return_value=None) as mock_save:
            policy.save("test_path")
            mock_save.assert_called_once_with("test_path")
            
        with patch.object(policy, 'load', return_value=None) as mock_load:
            policy.load("test_path")
            mock_load.assert_called_once_with("test_path")

    def test_multiple_agents(self):
        """Test building a policy with multiple independent agents."""
        policy = self.builder.add_component(
            component_id="agent1",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="agent2",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim + 2,  # Different obs dimension
            actor_obs_dim=self.obs_dim + 2,  # Different obs dimension
            num_actions=self.action_dim - 1,  # Different action dimension
            critic_out_dim=1,
            actor_hidden_dims=[32, 32],      # Different hidden dims
            critic_hidden_dims=[32, 32]      # Different hidden dims
        ).build()
        
        # Check if components exist
        self.assertIn("agent1", self.builder.components)
        self.assertIn("agent2", self.builder.components)
        
        # Test forward method with different input shapes
        batch_size = 5
        obs1 = torch.rand(batch_size, self.obs_dim).to(self.device)
        obs2 = torch.rand(batch_size, self.obs_dim + 2).to(self.device)
        
        obs = {
            "agent1": obs1,
            "agent2": obs2
        }
        
        # Get individual components
        agent1_component = policy.components["agent1"]
        agent2_component = policy.components["agent2"]
        
        # Test act method with deterministic flag
        agent1_out, agent1_info = agent1_component.act(obs1, deterministic=True)
        agent2_out, agent2_info = agent2_component.act(obs2, deterministic=True)
        policy_out, policy_info = policy.act(obs, deterministic=True)
        
        torch.testing.assert_close(agent1_out, policy_out["agent1"])
        torch.testing.assert_close(agent2_out, policy_out["agent2"])
        assert_dicts_close(agent1_info, policy_info["agent1"])
        assert_dicts_close(agent2_info, policy_info["agent2"])
        
        # Test evaluate method
        agent1_val = agent1_component.evaluate(obs1)
        agent2_val = agent2_component.evaluate(obs2)
        policy_val = policy.evaluate(obs)
        
        torch.testing.assert_close(agent1_val, policy_val["agent1"])
        torch.testing.assert_close(agent2_val, policy_val["agent2"])
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        agent1_out_stoch, agent1_info_stoch = agent1_component.act(obs1, deterministic=False)
        agent2_out_stoch, agent2_info_stoch = agent2_component.act(obs2, deterministic=False)
        
        torch.manual_seed(42)
        policy_out_stoch, policy_info_stoch = policy.act(obs, deterministic=False)
        
        torch.testing.assert_close(agent1_out_stoch, policy_out_stoch["agent1"])
        torch.testing.assert_close(agent2_out_stoch, policy_out_stoch["agent2"])
        assert_dicts_close(agent1_info_stoch, policy_info_stoch["agent1"])
        assert_dicts_close(agent2_info_stoch, policy_info_stoch["agent2"])
        
        # Mock the save and load methods for testing
        with patch.object(policy, 'save', return_value=None) as mock_save:
            policy.save("test_path")
            mock_save.assert_called_once_with("test_path")
            
        with patch.object(policy, 'load', return_value=None) as mock_load:
            policy.load("test_path")
            mock_load.assert_called_once_with("test_path")

    def test_varying_hidden_dims(self):
        """Test components with different hidden dimension configurations."""
        # Test each hidden dimension variation as a separate component
        for i, hidden_dims in enumerate(([64], [128, 64, 32], [256, 128])):
            component_id = f"agent{i}"
            
            # Using the same parameter structure as test_add_single_actor_critic
            policy = self.builder.add_component(
                component_id=component_id,
                network_type="mlp",
                network_class="actor_critic",
                critic_obs_dim=self.obs_dim,
                actor_obs_dim=self.obs_dim,
                num_actions=self.action_dim,
                critic_out_dim=1,
                actor_hidden_dims=hidden_dims,
                critic_hidden_dims=hidden_dims
            ).build()
            
            # Test act method
            batch_size = 5
            obs = torch.rand(batch_size, self.obs_dim).to(self.device)
            
            # Get the component network
            component = policy.components[component_id]
            
            # Test act method with deterministic flag
            component_out, component_info = component.act(obs, deterministic=True)
            policy_out, policy_info = policy.act({component_id: obs}, deterministic=True)

            # Verify policy outputs match direct component outputs
            torch.testing.assert_close(component_out, policy_out[component_id])
            assert_dicts_close(component_info, policy_info[component_id])

            # Test evaluate method
            component_out = component.evaluate(obs)
            policy_out = policy.evaluate({component_id: obs})

            torch.testing.assert_close(component_out, policy_out[component_id])
            
            # Test act method with stochastic sampling (set seed for reproducibility)
            torch.manual_seed(42)
            component_out_stoch, component_info_stoch = component.act(obs, deterministic=False)
            
            torch.manual_seed(42)
            policy_out_stoch, policy_info_stoch = policy.act({component_id: obs}, deterministic=False)
            
            torch.testing.assert_close(component_out_stoch, policy_out_stoch[component_id])
            assert_dicts_close(component_info_stoch, policy_info_stoch[component_id])
            
            # Mock the save and load methods for testing
            with patch.object(policy, 'save', return_value=None) as mock_save:
                policy.save("test_path")
                mock_save.assert_called_once_with("test_path")
                
            with patch.object(policy, 'load', return_value=None) as mock_load:
                policy.load("test_path")
                mock_load.assert_called_once_with("test_path")
            
            # Clear the builder for the next iteration
            self.builder = MultiAgentPolicyBuilder()

    def test_add_connection_between_components(self):
        """Test adding connections between components using an encoder."""
        policy = self.builder.add_component(
            component_id="encoder",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.obs_dim+self.action_dim,
            output_dim=self.obs_dim,  # Encoder output dimension
            hidden_dims=[32]
        ).add_component(
            component_id="agent",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,  # Extended to accommodate encoder input
            actor_obs_dim=self.obs_dim,  # Extended to accommodate encoder input
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_connection(
            source_id=["agent"],
            target_id="encoder",
            concat_dim=1  # Concatenate along feature dimension
        ).build()
        
        # Check if connection is properly set up
        self.assertIn("encoder", self.builder.connections)
        self.assertEqual(len(self.builder.connections["encoder"]), 1)
        self.assertEqual(self.builder.connections["encoder"][0]["source_id"], ["agent"])
        
        # Test with connected components
        batch_size = 5
        encoder_obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        agent_obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        
        obs = {
            "encoder": encoder_obs,
            "agent": agent_obs
        }
        
        # Get individual components
        encoder_component = policy.components["encoder"]
        agent_component = policy.components["agent"]
        
                
        # Manually create the concatenated input for agent
        agent_actions, _ = agent_component.act(obs["agent"], deterministic=True)
        concatenated_input = torch.cat([agent_actions, obs['encoder']], dim=1)
        
        # Test act method with deterministic flag
        encoder_out = encoder_component.forward(concatenated_input)
        policy_actions, _ = policy.act(obs, deterministic=True)
        
        torch.testing.assert_close(encoder_out, policy_actions["encoder"])
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        agent_actions_stoch, _ = agent_component.act(obs["agent"], deterministic=False)
        concatenated_input_stoch = torch.cat([agent_actions_stoch, obs['encoder']], dim=1)
        encoder_out_stoch = encoder_component.forward(concatenated_input_stoch)
        torch.manual_seed(42)
        policy_actions_stoch, _ = policy.act(obs, deterministic=False)
        torch.testing.assert_close(encoder_out_stoch, policy_actions_stoch["encoder"])
        
        
        # Test evaluate method
        agent_val = agent_component.evaluate(obs["agent"])
        policy_val = policy.evaluate(obs)
        
        torch.testing.assert_close(agent_val, policy_val["agent"])
        
        # Mock the save and load methods for testing
        with patch.object(policy, 'save', return_value=None) as mock_save:
            policy.save("test_path")
            mock_save.assert_called_once_with("test_path")
            
        with patch.object(policy, 'load', return_value=None) as mock_load:
            policy.load("test_path")
            mock_load.assert_called_once_with("test_path")

    def test_saving_and_loading(self):
        """Test the save and load functionality."""
        # Build a simple policy with updated parameter names matching test_add_single_actor_critic
        policy = self.builder.add_component(
            component_id="agent1",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).build()
        
        # Create a batch of observations for testing
        batch_size = 5
        obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        
        # Get initial outputs before saving using act and evaluate (matching test_add_single_actor_critic)
        torch.manual_seed(42)
        actions1, info1 = policy.act({"agent1": obs}, deterministic=True)
        values1 = policy.evaluate({"agent1": obs})
        
        # Save the model
        save_path = "test_model"
        policy.save(save_path)
        
        # Modify the weights to simulate a change
        params = policy.parameters()
        for key, param in params.items():
            actor_params = param["actor"]
            critic_params = param["critic"]
            for data in actor_params:
                data.data = data.data * 2.0
            for data in critic_params:
                data.data = data.data * 2.0
            
        # Verify outputs are different after weight modification
        torch.manual_seed(42)
        actions2, info2 = policy.act({"agent1": obs}, deterministic=True)
        values2 = policy.evaluate({"agent1": obs})
        
        # The outputs should be different now
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actions1["agent1"], actions2["agent1"])
            torch.testing.assert_close(values1["agent1"], values2["agent1"])
            assert_dicts_close(info1["agent1"], info2["agent1"])
            
        # Load the saved weights
        policy.load(save_path)
        
        # Verify outputs are restored to initial values
        torch.manual_seed(42)
        actions_restored, info_restored = policy.act({"agent1": obs}, deterministic=True)
        values_restored = policy.evaluate({"agent1": obs})
        
        # Assert that the outputs match the original values
        torch.testing.assert_close(actions1["agent1"], actions_restored["agent1"])
        torch.testing.assert_close(values1["agent1"], values_restored["agent1"])
        assert_dicts_close(info1["agent1"], info_restored["agent1"])
        
        # Also test stochastic actions
        torch.manual_seed(42)
        stoch_actions1, stoch_info1 = policy.act({"agent1": obs}, deterministic=False)
        
        # Modify weights again
        for key, param in params.items():
            for data in param["actor"]:
                data.data = data.data * 1.5
            for data in param["critic"]:
                data.data = data.data * 1.5
        
        # Verify different outputs after modification
        torch.manual_seed(42)
        stoch_actions2, stoch_info2 = policy.act({"agent1": obs}, deterministic=False)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(stoch_actions1["agent1"], stoch_actions2["agent1"])
        
        # Load the saved weights again
        policy.load(save_path)
        
        # Verify stochastic actions are also restored
        torch.manual_seed(42)
        stoch_actions_restored, stoch_info_restored = policy.act({"agent1": obs}, deterministic=False)
        torch.testing.assert_close(stoch_actions1["agent1"], stoch_actions_restored["agent1"])
        assert_dicts_close(stoch_info1["agent1"], stoch_info_restored["agent1"])

    def test_complex_connection_chain(self):
        """Test a complex chain of connections between multiple components with encoders after actors."""
        policy = self.builder.add_component(
            component_id="agent1",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="agent2",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="agent3",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,
            actor_obs_dim=self.obs_dim,
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="encoder1",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.action_dim * 2,  # Taking input from agent1 and agent2 actions
            output_dim=self.obs_dim // 2,
            hidden_dims=[48, 24]
        ).add_component(
            component_id="encoder2",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.action_dim * 2,  # Taking input from agent2 and agent3 actions
            output_dim=self.obs_dim // 2,
            hidden_dims=[48, 24]
        ).add_component(
            component_id="final_encoder",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.obs_dim,  # Taking input from encoder1 and encoder2 outputs
            output_dim=self.obs_dim,
            hidden_dims=[64, 32]
        ).add_connection(
            source_id=["agent1", "agent2"],
            target_id="encoder1",
            concat_dim=1  # Concatenate along feature dimension
        ).add_connection(
            source_id=["agent2", "agent3"],
            target_id="encoder2",
            concat_dim=1  # Concatenate along feature dimension
        ).add_connection(
            source_id=["encoder1", "encoder2"],
            target_id="final_encoder",
            concat_dim=1  # Concatenate along feature dimension
        ).build()
        
        # Check if connections are properly set up
        for target_id, sources in [
            ("encoder1", ["agent1", "agent2"]),
            ("encoder2", ["agent2", "agent3"]),
            ("final_encoder", ["encoder1", "encoder2"])
        ]:
            self.assertIn(target_id, self.builder.connections)
            connection = next(conn for conn in self.builder.connections[target_id] 
                            if set(conn["source_id"]) == set(sources))
            self.assertEqual(set(connection["source_id"]), set(sources))
        
        # Test with connected components
        batch_size = 5
        obs = {
            "agent1": torch.rand(batch_size, self.obs_dim).to(self.device),
            "agent2": torch.rand(batch_size, self.obs_dim).to(self.device),
            "agent3": torch.rand(batch_size, self.obs_dim).to(self.device)
        }
        
        # Get individual components
        agent1 = policy.components["agent1"]
        agent2 = policy.components["agent2"]
        agent3 = policy.components["agent3"]
        encoder1 = policy.components["encoder1"]
        encoder2 = policy.components["encoder2"]
        final_encoder = policy.components["final_encoder"]
        
        # DETERMINISTIC TESTING
        
        # 1. Get actions from agent networks
        torch.manual_seed(42)
        agent1_actions, _ = agent1.act(obs["agent1"], deterministic=True)
        agent2_actions, _ = agent2.act(obs["agent2"], deterministic=True)
        agent3_actions, _ = agent3.act(obs["agent3"], deterministic=True)
        
        # 2. Feed actions to encoder1 and encoder2
        encoder1_input = torch.cat([agent1_actions, agent2_actions], dim=1)
        encoder2_input = torch.cat([agent2_actions, agent3_actions], dim=1)
        
        encoder1_output = encoder1.forward(encoder1_input)
        encoder2_output = encoder2.forward(encoder2_input)
        
        # 3. Feed encoder outputs to final encoder
        final_input = torch.cat([encoder1_output, encoder2_output], dim=1)
        final_output = final_encoder.forward(final_input)
        
        # 4. Get outputs through the policy network
        torch.manual_seed(42)
        policy_outputs, _ = policy.act(obs, deterministic=True)
        
        # Compare outputs
        torch.testing.assert_close(encoder1_output, policy_outputs["encoder1"])
        torch.testing.assert_close(encoder2_output, policy_outputs["encoder2"])
        torch.testing.assert_close(final_output, policy_outputs["final_encoder"])
        torch.testing.assert_close(agent1_actions, policy_outputs["agent1"])
        torch.testing.assert_close(agent2_actions, policy_outputs["agent2"])
        torch.testing.assert_close(agent3_actions, policy_outputs["agent3"])
        
        # STOCHASTIC TESTING
        
        # 1. Get actions from agent networks
        torch.manual_seed(42)
        agent1_actions_stoch, _ = agent1.act(obs["agent1"], deterministic=False)
        agent2_actions_stoch, _ = agent2.act(obs["agent2"], deterministic=False)
        agent3_actions_stoch, _ = agent3.act(obs["agent3"], deterministic=False)
        
        # 2. Feed actions to encoder1 and encoder2
        encoder1_input_stoch = torch.cat([agent1_actions_stoch, agent2_actions_stoch], dim=1)
        encoder2_input_stoch = torch.cat([agent2_actions_stoch, agent3_actions_stoch], dim=1)
        
        encoder1_output_stoch = encoder1.forward(encoder1_input_stoch)
        encoder2_output_stoch = encoder2.forward(encoder2_input_stoch)
        
        # 3. Feed encoder outputs to final encoder
        final_input_stoch = torch.cat([encoder1_output_stoch, encoder2_output_stoch], dim=1)
        final_output_stoch = final_encoder.forward(final_input_stoch)
        
        # 4. Get outputs through the policy network
        torch.manual_seed(42)
        policy_outputs_stoch, _ = policy.act(obs, deterministic=False)
        
        # Compare outputs
        torch.testing.assert_close(encoder1_output_stoch, policy_outputs_stoch["encoder1"])
        torch.testing.assert_close(encoder2_output_stoch, policy_outputs_stoch["encoder2"])
        torch.testing.assert_close(final_output_stoch, policy_outputs_stoch["final_encoder"])
        torch.testing.assert_close(agent1_actions_stoch, policy_outputs_stoch["agent1"])
        torch.testing.assert_close(agent2_actions_stoch, policy_outputs_stoch["agent2"])
        torch.testing.assert_close(agent3_actions_stoch, policy_outputs_stoch["agent3"])
        
        # Test evaluate method
        agent1_val = agent1.evaluate(obs["agent1"])
        agent2_val = agent2.evaluate(obs["agent2"])
        agent3_val = agent3.evaluate(obs["agent3"])
        policy_val = policy.evaluate(obs)
        
        torch.testing.assert_close(agent1_val, policy_val["agent1"])
        torch.testing.assert_close(agent2_val, policy_val["agent2"])
        torch.testing.assert_close(agent3_val, policy_val["agent3"])
        
        # Mock the save and load methods for testing
        with patch.object(policy, 'save', return_value=None) as mock_save:
            policy.save("test_path")
            mock_save.assert_called_once_with("test_path")
            
        with patch.object(policy, 'load', return_value=None) as mock_load:
            policy.load("test_path")
            mock_load.assert_called_once_with("test_path")


def assert_dicts_close(dict1, dict2, rtol=1e-5, atol=1e-8):
    """Compare two dictionaries of tensors for equality."""
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"
    for key in dict1:
        torch.testing.assert_close(dict1[key], dict2[key], rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()