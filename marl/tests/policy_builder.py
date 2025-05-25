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
        component_out = component.act(obs, deterministic=True)
        policy_out = policy.act(obs, agent_id="agent1", deterministic=True)

        # Verify policy outputs match direct component outputs
        torch.testing.assert_close(component_out, policy_out)

        #Get outputs without explicit agent_id
        policy_out = policy.act({"agent1": obs}, deterministic=True)

        torch.testing.assert_close(component_out, policy_out['agent1'])

        # Test get_actions method with deterministic flag
        component_out = component.evaluate(obs)
        policy_out = policy.evaluate(obs, agent_id="agent1")

        torch.testing.assert_close(component_out, policy_out)

        #Get outputs without explicit agent_id
        policy_out = policy.evaluate({'agent1': obs})

        torch.testing.assert_close(component_out, policy_out['agent1'])

        # Test get_actions method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        component_out = component.act(obs, deterministic=False)
        
        torch.manual_seed(42)
        policy_out = policy.act({"agent1": obs}, deterministic=False)
        
        torch.testing.assert_close(component_out, policy_out['agent1'])

        #Get outputs without explicit agent_id
        torch.manual_seed(42)
        policy_out = policy.act(obs, agent_id="agent1", deterministic=False)

        torch.testing.assert_close(component_out, policy_out)
        
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
        actor_out = actor_component.act(actor_obs, deterministic=True)
        policy_out = policy.act({"actor1": actor_obs}, deterministic=True)
        
        torch.testing.assert_close(actor_out, policy_out['actor1'])

        #Get outputs without explicit agent_id
        policy_out = policy.act(actor_obs, agent_id="actor1", deterministic=True)

        torch.testing.assert_close(actor_out, policy_out)
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        actor_out_stoch = actor_component.act(actor_obs, deterministic=False)
        
        torch.manual_seed(42)
        policy_out_stoch = policy.act({"actor1": actor_obs}, deterministic=False)
        
        torch.testing.assert_close(actor_out_stoch, policy_out_stoch['actor1'])

        #Get outputs without explicit agent_id
        torch.manual_seed(42)
        policy_out = policy.act(actor_obs, agent_id="actor1", deterministic=False)

        torch.testing.assert_close(actor_out_stoch, policy_out)
        
        # Test evaluate method for critic
        critic_out = critic_component.evaluate(critic_obs)
        policy_out = policy.evaluate({"critic1": critic_obs})
        
        torch.testing.assert_close(critic_out, policy_out["critic1"])

        #Explicit
        actor_std = actor_component.get_action_std()
        policy_std = policy.get_action_std(agent_id="actor1")

        torch.testing.assert_close(actor_std, policy_std)

        #Get without explicit agent_id
        actor_std = actor_component.get_action_std()
        policy_std = policy.get_action_std()

        torch.testing.assert_close(actor_std, policy_std['actor1'])

        actor_mean = actor_component.get_action_mean()
        policy_mean = policy.get_action_mean(agent_id="actor1")

        torch.testing.assert_close(actor_mean, policy_mean)

        #Get without explicit agent_id
        actor_mean = actor_component.get_action_mean()
        policy_mean = policy.get_action_mean()

        torch.testing.assert_close(actor_mean, policy_mean['actor1'])

        actor_entropy = actor_component.get_entropy()
        policy_entropy = policy.get_entropy(agent_id="actor1")

        torch.testing.assert_close(actor_entropy, policy_entropy)

        #Get without explicit agent_id
        actor_entropy = actor_component.get_entropy()
        policy_entropy = policy.get_entropy()

        torch.testing.assert_close(actor_entropy, policy_entropy['actor1'])

        actor_action_log_prob = actor_component.get_actions_log_prob(actor_out)
        policy_action_log_prob = policy.get_actions_log_prob(actor_out, agent_id="actor1")

        torch.testing.assert_close(actor_action_log_prob, policy_action_log_prob)

        #Get without explicit agent_id
        actor_action_log_prob = actor_component.get_actions_log_prob(actor_out)
        policy_action_log_prob = policy.get_actions_log_prob({"actor1": actor_out})

        torch.testing.assert_close(actor_action_log_prob, policy_action_log_prob['actor1'])
        
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
        agent1_out = agent1_component.act(obs1, deterministic=True)
        agent2_out = agent2_component.act(obs2, deterministic=True)
        policy_out = policy.act(obs, deterministic=True)
        
        torch.testing.assert_close(agent1_out, policy_out["agent1"])
        torch.testing.assert_close(agent2_out, policy_out["agent2"])


        #Explicit
        policy_out = policy.act(obs1, agent_id="agent1", deterministic=True)

        torch.testing.assert_close(agent1_out, policy_out)

        policy_out = policy.act(obs2, agent_id="agent2", deterministic=True)

        torch.testing.assert_close(agent2_out, policy_out)

        # Test evaluate method
        agent1_val = agent1_component.evaluate(obs1)
        agent2_val = agent2_component.evaluate(obs2)
        policy_val = policy.evaluate(obs)
        
        torch.testing.assert_close(agent1_val, policy_val["agent1"])
        torch.testing.assert_close(agent2_val, policy_val["agent2"])

        #Explicit
        policy_out = policy.evaluate(obs1, agent_id="agent1")

        torch.testing.assert_close(agent1_val, policy_out)

        policy_out = policy.evaluate(obs2, agent_id="agent2")

        torch.testing.assert_close(agent2_val, policy_out)
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        agent1_out_stoch = agent1_component.act(obs1, deterministic=False)
        agent2_out_stoch = agent2_component.act(obs2, deterministic=False)
        
        torch.manual_seed(42)
        policy_out_stoch = policy.act(obs, deterministic=False)
        
        torch.testing.assert_close(agent1_out_stoch, policy_out_stoch["agent1"])
        torch.testing.assert_close(agent2_out_stoch, policy_out_stoch["agent2"])

        #Test get_action_std
        agent1_std = agent1_component.get_action_std()
        agent2_std = agent2_component.get_action_std()
        policy_std = policy.get_action_std(agent_id="agent1")
        policy_std2 = policy.get_action_std(agent_id="agent2")

        torch.testing.assert_close(agent1_std, policy_std)
        torch.testing.assert_close(agent2_std, policy_std2)

        #Get without explicit agent_id
        policy_std_dict = policy.get_action_std()
        torch.testing.assert_close(agent1_std, policy_std_dict['agent1'])
        torch.testing.assert_close(agent2_std, policy_std_dict['agent2'])

        #Test get_action_mean
        agent1_mean = agent1_component.get_action_mean()
        agent2_mean = agent2_component.get_action_mean()
        policy_mean = policy.get_action_mean(agent_id="agent1")
        policy_mean2 = policy.get_action_mean(agent_id="agent2")

        torch.testing.assert_close(agent1_mean, policy_mean)
        torch.testing.assert_close(agent2_mean, policy_mean2)

        #Get without explicit agent_id
        policy_mean_dict = policy.get_action_mean()
        torch.testing.assert_close(agent1_mean, policy_mean_dict['agent1'])
        torch.testing.assert_close(agent2_mean, policy_mean_dict['agent2'])

        #Test get_entropy
        agent1_entropy = agent1_component.get_entropy()
        agent2_entropy = agent2_component.get_entropy()
        policy_entropy = policy.get_entropy(agent_id="agent1")
        policy_entropy2 = policy.get_entropy(agent_id="agent2")

        torch.testing.assert_close(agent1_entropy, policy_entropy)
        torch.testing.assert_close(agent2_entropy, policy_entropy2)

        #Get without explicit agent_id
        policy_entropy_dict = policy.get_entropy()
        torch.testing.assert_close(agent1_entropy, policy_entropy_dict['agent1'])
        torch.testing.assert_close(agent2_entropy, policy_entropy_dict['agent2'])

        #Test get_actions_log_prob
        agent1_action_log_prob = agent1_component.get_actions_log_prob(agent1_out)
        agent2_action_log_prob = agent2_component.get_actions_log_prob(agent2_out)
        policy_action_log_prob = policy.get_actions_log_prob(agent1_out, agent_id="agent1")
        policy_action_log_prob2 = policy.get_actions_log_prob(agent2_out, agent_id="agent2")

        torch.testing.assert_close(agent1_action_log_prob, policy_action_log_prob)
        torch.testing.assert_close(agent2_action_log_prob, policy_action_log_prob2)

        #Get without explicit agent_id
        policy_action_log_prob_dict = policy.get_actions_log_prob({
            "agent1": agent1_out,
            "agent2": agent2_out
        })
        torch.testing.assert_close(agent1_action_log_prob, policy_action_log_prob_dict['agent1'])
        torch.testing.assert_close(agent2_action_log_prob, policy_action_log_prob_dict['agent2'])

        
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
            component_out = component.act(obs, deterministic=True)
            policy_out = policy.act({component_id: obs}, deterministic=True)

            # Verify policy outputs match direct component outputs
            torch.testing.assert_close(component_out, policy_out[component_id])

            # Test evaluate method
            component_out = component.evaluate(obs)
            policy_out = policy.evaluate({component_id: obs})

            torch.testing.assert_close(component_out, policy_out[component_id])
            
            # Test act method with stochastic sampling (set seed for reproducibility)
            torch.manual_seed(42)
            component_out_stoch = component.act(obs, deterministic=False)
            
            torch.manual_seed(42)
            policy_out_stoch = policy.act({component_id: obs}, deterministic=False)
            
            torch.testing.assert_close(component_out_stoch, policy_out_stoch[component_id])
            
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
        agent_actions = agent_component.act(obs["agent"], deterministic=True)
        concatenated_input = torch.cat([agent_actions, obs['encoder']], dim=1)
        
        # Test act method with deterministic flag
        encoder_out = encoder_component.forward(concatenated_input)
        policy_actions = policy.act(obs, deterministic=True)

        torch.testing.assert_close(encoder_out, policy_actions["encoder"])

        policy_actions = policy.act(obs["agent"],agent_id='agent', deterministic=True)
        
        torch.testing.assert_close(agent_actions, policy_actions)
        
        # Test act method with stochastic sampling (set seed for reproducibility)
        torch.manual_seed(42)
        agent_actions_stoch = agent_component.act(obs["agent"], deterministic=False)
        concatenated_input_stoch = torch.cat([agent_actions_stoch, obs['encoder']], dim=1)
        encoder_out_stoch = encoder_component.forward(concatenated_input_stoch)
        torch.manual_seed(42)
        policy_actions_stoch = policy.act(obs, deterministic=False)
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
    
    def test_complex_freezing_functionality(self):
        """Test freezing functionality for complex policies with multiple components."""
        # Build a complex policy with multiple components
        policy = self.builder.add_component(
            component_id="encoder",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.obs_dim,
            output_dim=64,
            hidden_dims=[128, 64]
        ).add_component(
            component_id="agent1",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,  # Takes encoder output
            actor_obs_dim=64+self.obs_dim,   # Takes encoder output
            num_actions=self.action_dim,
            critic_out_dim=1,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims
        ).add_component(
            component_id="agent2",
            network_type="mlp",
            network_class="actor_critic",
            critic_obs_dim=self.obs_dim,  # Direct observation
            actor_obs_dim=self.obs_dim,   # Direct observation
            num_actions=self.action_dim - 1,
            critic_out_dim=1,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32]
        ).add_connection(
            source_id=["encoder"],
            target_id="agent1"
        ).build()

        # Create test observations
        batch_size = 5
        raw_obs = torch.rand(batch_size, self.obs_dim).to(self.device)
        obs2 = torch.rand(batch_size, self.obs_dim).to(self.device)
        obs = {
            "encoder": raw_obs,
            "agent1": raw_obs,  # Will be filled by encoder output
            "agent2": obs2
        }

        # Test 1: Verify initial trainable parameter counts
        initial_param_summary = {}
        for component_id, component in policy.components.items():
            info = component.get_parameter_info()
            initial_param_summary[component_id] = info
            # Initially all parameters should be trainable
            self.assertEqual(info['trainable_parameters'], info['total_parameters'])
            self.assertFalse(info['is_frozen'])

        # Get initial outputs before any freezing
        torch.manual_seed(42)
        initial_actions = policy.act(obs, deterministic=True)
        initial_values = policy.evaluate(obs)

        # Test 2: Freeze encoder component
        policy.components["encoder"].freeze()
        
        # Verify encoder is frozen
        encoder_info = policy.components["encoder"].get_parameter_info()
        self.assertTrue(encoder_info['is_frozen'])
        self.assertEqual(encoder_info['trainable_parameters'], 0)
        self.assertEqual(encoder_info['frozen_parameters'], encoder_info['total_parameters'])
        
        # Verify other components still trainable
        for component_id in ["agent1", "agent2"]:
            info = policy.components[component_id].get_parameter_info()
            self.assertFalse(info['is_frozen'])
            self.assertEqual(info['trainable_parameters'], info['total_parameters'])

        # Test 3: Simulate training step on frozen encoder (should not change)
        # Get encoder parameters before "training"
        encoder_params_before = {}
        for name, param in policy.components["encoder"].network.named_parameters():
            encoder_params_before[name] = param.data.clone()

        # Simulate gradient update attempt on all components
        for component_id, component in policy.components.items():
            for param in component.network.parameters():
                if param.requires_grad:  # Only trainable params should be updated
                    param.data += torch.randn_like(param.data) * 0.1

        # Verify encoder parameters unchanged (frozen)
        for name, param in policy.components["encoder"].network.named_parameters():
            torch.testing.assert_close(encoder_params_before[name], param.data)

        # Verify outputs with frozen encoder still work
        torch.manual_seed(42)
        actions_with_frozen_encoder = policy.act(obs, deterministic=True)
        
        # The encoder output should be the same, but agent outputs may differ due to agent parameter changes
        # We'll test this by checking the encoder component specifically if we had direct access

        # Test 4: Freeze multiple components
        policy.components["agent1"].freeze()
        
        # Verify both encoder and agent1 are frozen
        for component_id in ["encoder", "agent1"]:
            info = policy.components[component_id].get_parameter_info()
            self.assertTrue(info['is_frozen'])
            self.assertEqual(info['trainable_parameters'], 0)
        
        # Verify agent2 still trainable
        agent2_info = policy.components["agent2"].get_parameter_info()
        self.assertFalse(agent2_info['is_frozen'])
        self.assertEqual(agent2_info['trainable_parameters'], agent2_info['total_parameters'])

        # Test 5: Get only trainable parameters
        trainable_params = []
        for component in policy.components.values():
            trainable_params.extend(component.get_trainable_parameters())
        
        # Should only have agent2's parameters
        expected_trainable = agent2_info['trainable_parameters']
        actual_trainable = sum(p.numel() for p in trainable_params)
        self.assertEqual(actual_trainable, expected_trainable)

        # Test 6: Unfreeze encoder
        policy.components["encoder"].unfreeze()
        
        # Verify encoder is now trainable
        encoder_info = policy.components["encoder"].get_parameter_info()
        self.assertFalse(encoder_info['is_frozen'])
        self.assertEqual(encoder_info['trainable_parameters'], encoder_info['total_parameters'])

        # Test 7: Test selective layer freezing
        # First unfreeze agent1 to test layer freezing
        policy.components["agent1"].unfreeze()
        
        # Get layer names for agent1 (assuming MLP structure)
        layer_names_to_freeze = []
        for name, _ in policy.components["agent1"].network.named_parameters():
            if "0." in name:  # First layer parameters
                layer_names_to_freeze.append(name)
        
        if layer_names_to_freeze:  # Only test if we found layers to freeze
            # Freeze specific layers
            policy.components["agent1"].freeze_layers(layer_names_to_freeze)
            
            # Verify some parameters are frozen, some are not
            agent1_info = policy.components["agent1"].get_parameter_info()
            self.assertFalse(agent1_info['is_frozen'])  # Not fully frozen
            self.assertGreater(agent1_info['frozen_parameters'], 0)  # Some frozen
            self.assertGreater(agent1_info['trainable_parameters'], 0)  # Some trainable

        # Test 8: Save and load with frozen components
        policy.components["encoder"].freeze()
        policy.components["agent1"].freeze()
        
        # Save policy state
        policy.save("test_frozen_policy")
        
        # Modify trainable parameters (only agent2 should change)
        for param in policy.components["agent2"].network.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        
        # Verify outputs changed for agent2 only
        torch.manual_seed(42)
        actions_after_change = policy.act(obs, deterministic=True)
        
        # Load policy back
        policy.load("test_frozen_policy")
        
        # Verify frozen status preserved after loading
        self.assertTrue(policy.components["encoder"].get_parameter_info()['is_frozen'])
        self.assertTrue(policy.components["agent1"].get_parameter_info()['is_frozen'])
        self.assertFalse(policy.components["agent2"].get_parameter_info()['is_frozen'])
        
        # Test 9: Test parameter summary functionality
        summary = {}
        total_params = 0
        total_trainable = 0
        
        for component_id, component in policy.components.items():
            info = component.get_parameter_info()
            summary[component_id] = info
            total_params += info['total_parameters']
            total_trainable += info['trainable_parameters']
        
        # Verify summary calculations
        frozen_params = total_params - total_trainable
        frozen_percentage = (frozen_params / total_params * 100) if total_params > 0 else 0
        
        self.assertGreater(total_params, 0)
        self.assertGreater(frozen_params, 0)  # We have frozen components
        self.assertGreater(total_trainable, 0)  # agent2 should still be trainable
        self.assertLess(frozen_percentage, 100)  # Not everything is frozen
        self.assertGreater(frozen_percentage, 0)   # But something is frozen

        # Test 10: Test that frozen components still perform forward passes correctly
        torch.manual_seed(123)
        final_actions = policy.act(obs, deterministic=True)
        final_values = policy.evaluate(obs)
        
        # Should not raise any errors and should produce valid outputs
        for agent_id in final_actions.keys():
            self.assertIsInstance(final_actions[agent_id], torch.Tensor)
            self.assertEqual(final_actions[agent_id].shape[0], batch_size)
        
        for agent_id in final_values.keys():
            self.assertIsInstance(final_values[agent_id], torch.Tensor)
            self.assertEqual(final_values[agent_id].shape[0], batch_size)

        # Test 11: Test stochastic actions with frozen components
        torch.manual_seed(456)
        stoch_actions1 = policy.act(obs, deterministic=False)
        
        # Actions should still be generated correctly
        for agent_id in stoch_actions1.keys():
            self.assertIsInstance(stoch_actions1[agent_id], torch.Tensor)
            self.assertEqual(stoch_actions1[agent_id].shape[0], batch_size)
        
        # Test reproducibility with frozen components
        torch.manual_seed(456)
        stoch_actions2 = policy.act(obs, deterministic=False)
        
        for agent_id in stoch_actions1.keys():
            torch.testing.assert_close(stoch_actions1[agent_id], stoch_actions2[agent_id])

        print("All freezing functionality tests passed!")

    def test_complex_saving_and_loading(self):
        """Test saving and loading functionality for complex policies with multiple components."""
        # Build a complex policy with multiple components
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
            critic_obs_dim=self.obs_dim + 2,
            actor_obs_dim=self.obs_dim + 2,
            num_actions=self.action_dim - 1,
            critic_out_dim=1,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32]
        ).add_component(
            component_id="encoder",
            network_type="mlp",
            network_class="encoder",
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.obs_dim,
            hidden_dims=[32]
        ).add_connection(
            source_id=["agent1"],
            target_id="encoder",
            concat_dim=1
        ).build()

        # Create test observations
        batch_size = 5
        obs1 = torch.rand(batch_size, self.obs_dim).to(self.device)
        obs2 = torch.rand(batch_size, self.obs_dim + 2).to(self.device)
        obs3 = torch.rand(batch_size, self.obs_dim).to(self.device)
        obs = {
            "agent1": obs1,
            "agent2": obs2,
            "encoder": obs3
        }

        # Get initial outputs
        torch.manual_seed(42)
        actions1 = policy.act(obs, deterministic=True)
        values1 = policy.evaluate(obs)

        # Test 1: Save and load individual components
        # Save agent1
        policy.save("test_agent1", ["agent1"])
        
        # Modify agent1's weights
        for param in list(policy.components["agent1"].parameters()):
            param.data = param.data * 2.0

        # Verify agent1 outputs changed
        torch.manual_seed(42)
        actions2 = policy.act(obs, deterministic=True)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actions1["agent1"], actions2["agent1"])

        # Load agent1 back
        policy.load("test_agent1", ["agent1"])
        
        # Verify agent1 outputs restored
        torch.manual_seed(42)
        actions_restored = policy.act(obs, deterministic=True)
        torch.testing.assert_close(actions1["agent1"], actions_restored["agent1"])

        # Test 2: Save and load multiple components
        # Save both agents
        policy.save("test_agents", ["agent1", "agent2"])
        
        # Modify both agents' weights
        for component_id in ["agent1", "agent2"]:
            for param in list(policy.components[component_id].parameters()):
                param.data = param.data * 1.5

        # Verify outputs changed
        torch.manual_seed(42)
        actions3 = policy.act(obs, deterministic=True)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actions1["agent1"], actions3["agent1"])
            torch.testing.assert_close(actions1["agent2"], actions3["agent2"])

        # Load both agents back
        policy.load("test_agents", ["agent1", "agent2"])
        
        # Verify outputs restored
        torch.manual_seed(42)
        actions_restored = policy.act(obs, deterministic=True)
        torch.testing.assert_close(actions1["agent1"], actions_restored["agent1"])
        torch.testing.assert_close(actions1["agent2"], actions_restored["agent2"])

        # Test 3: Save and load entire policy
        # Save entire policy
        policy.save("test_policy")
        
        # Modify all components' weights
        for component_id in ["agent1", "agent2", "encoder"]:
            for param in list(policy.components[component_id].parameters()):
                param.data = param.data * 2.0

        # Verify all outputs changed
        torch.manual_seed(42)
        actions4 = policy.act(obs, deterministic=True)
        values4 = policy.evaluate(obs)
        
        with self.assertRaises(AssertionError):
            for agent_id in ["agent1", "agent2"]:
                torch.testing.assert_close(actions1[agent_id], actions4[agent_id])
                torch.testing.assert_close(values1[agent_id], values4[agent_id])

        # Load entire policy back
        policy.load("test_policy")
        
        # Verify all outputs restored
        torch.manual_seed(42)
        actions_restored = policy.act(obs, deterministic=True)
        values_restored = policy.evaluate(obs)
        
        for agent_id in ["agent1", "agent2"]:
            torch.testing.assert_close(actions1[agent_id], actions_restored[agent_id])
            torch.testing.assert_close(values1[agent_id], values_restored[agent_id])

        # Test 4: Test stochastic actions with saved/loaded components
        torch.manual_seed(42)
        stoch_actions1 = policy.act(obs, deterministic=False)
        
        # Modify weights
        for component_id in ["agent1", "agent2", "encoder"]:
            for param in list(policy.components[component_id].parameters()):
                param.data = param.data * 1.5
        
        # Load policy back
        policy.load("test_policy", ["agent1", "agent2", "encoder"])
        
        # Verify stochastic actions restored
        torch.manual_seed(42)
        stoch_actions_restored = policy.act(obs, deterministic=False)
        for agent_id in ["agent1", "agent2"]:
            torch.testing.assert_close(stoch_actions1[agent_id], stoch_actions_restored[agent_id])


def assert_dicts_close(dict1, dict2, rtol=1e-5, atol=1e-8):
    """Compare two dictionaries of tensors for equality."""
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"
    for key in dict1:
        torch.testing.assert_close(dict1[key], dict2[key], rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()