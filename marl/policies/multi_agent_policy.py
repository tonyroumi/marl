import torch
from typing import Dict, Any, List, Optional, Union
from marl.policies import BasePolicy
from marl.policies.component import Component
import os

class MultiAgentPolicy(BasePolicy):
    """
    A composite multi-agent policy that orchestrates multiple interconnected components.
    
    This class manages a collection of neural network components (actors, critics, encoders) and their
    data flow connections to create complex multi-agent systems. It supports:
    - Individual agent architectures with different network types and roles
    - Inter-component communication through configurable connections
    - Selective agent processing for efficient inference

    
    Raises:
        ValueError: If component dependencies form cycles during validation.
        ValueError: If connection references non-existent components during validation.
        ValueError: If attempting to get actions from components without actor networks.
        ValueError: If attempting to evaluate components without critic networks.
        ValueError: If specified agent_id not found in components during method calls.
        ValueError: If source component outputs not available during execution.
        ValueError: If required actions not provided for log probability calculations.
    """
    def __init__(
        self, 
        components: Dict[str, Component],
        connections: Dict[str, Dict[str, Any]],
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.components = components
        self.connections = connections

        #Validate the policy structure
        self._validate_policy_structure()
    
    
    def act(
        self, 
        obs, 
        deterministic: bool = False, 
        agent_id: Optional[str] = None, 
        return_all: bool = True
        ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get actions from policy components.
        
        Args:
            obs: Observations - tensor if agent_id specified, dict of {agent_id: tensor} if not
            deterministic: Whether to use deterministic actions
            agent_id: Specific agent ID to get actions for. If None, returns all agents.
        
        Returns:
            If agent_id is specified: Action tensor for that agent
            If agent_id is None: Dict of {'agent_id': action_tensor} for all actor components
        """
        # If agent_id is specified, validate it exists
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
        
        component_outputs = {}
        execution_order = self._determine_execution_order()
        
        # Filter execution order if specific agent_id is requested
        if agent_id is not None:
            # Only process the specified agent and its dependencies
            filtered_order = self._get_agent_dependencies(agent_id, execution_order)
        else:
            filtered_order = execution_order
        
        for component_id in filtered_order:
            # Skip if agent_id is specified and this isn't the target agent or its dependency
            if agent_id is not None and not self._is_required_for_agent(component_id, agent_id):
                continue
                
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not found in components")
            
            component = self.components[component_id]
            
            # Handle connected components
            if component_id in self.connections:
                for conn in self.connections[component_id]:
                    source_ids = conn['source_id']
                    concat_dim = conn['concat_dim']
                    target_inputs = []
                    for source_id in source_ids:
                        if source_id not in component_outputs:
                            raise ValueError(f"Source {source_id} not found in component outputs")
                        target_inputs.append(component_outputs[source_id])
                    target_inputs = torch.cat(target_inputs, dim=concat_dim) if len(target_inputs) > 1 else target_inputs[0]
                    
                    # Add direct observations if available for this component
                    if agent_id is not None:
                        # Single agent mode - check if this component has direct obs access
                        if hasattr(component, 'uses_direct_obs') and component.uses_direct_obs:
                            target_inputs = torch.cat([target_inputs, obs], dim=concat_dim)
                    else:
                        # Multi-agent mode - check if this component has obs in the dict
                        if component_id in obs:
                            target_inputs = torch.cat([target_inputs, obs[component_id]], dim=concat_dim)
                    
                    # Process based on component type
                    if component.network_class in ["actor", "actor_critic"]:
                        component_outputs[component_id] = component.act(target_inputs, deterministic=deterministic)
                    elif component.network_class == "encoder":
                        component_outputs[component_id] = component.forward(target_inputs)
            else:
                # Handle non-connected components
                if component.network_class in ["actor", "actor_critic"]:
                    # Get the appropriate observation for this component
                    if agent_id is not None:
                        if type(obs) == dict:
                            raise ValueError(f"Observations must be a tensor for single agent mode. (no dependencies)")
                        # Single agent mode - obs is a tensor
                        if component_id == agent_id:
                            return component.act(obs, deterministic=deterministic)
                    else:
                        # Multi-agent mode - obs is a dict
                        if type(obs) != dict:
                            raise ValueError(f"No specific observations provided. Must specify agent_id or define agent specific observations..")
                        if component_id in obs:
                            component_outputs[component_id] = component.act(obs[component_id], deterministic=deterministic)
                        else:
                            raise ValueError(f"No observation provided for agent {component_id}")
                elif component.network_class != "critic":
                    # For encoders, handle obs appropriately
                    if agent_id is not None:
                        component_outputs[component_id] = component.forward(obs)
                    else:
                        # In multi-agent mode, encoder might need specific obs or all obs
                        if component_id in obs:
                            component_outputs[component_id] = component.forward(obs[component_id])
                        else:
                            raise ValueError(f"No observation provided for {component_id}")
        
        # If specific agent was requested, return its output
        if agent_id is not None:
            if agent_id in component_outputs:
                return component_outputs[agent_id]
            else:
                raise ValueError(f"No output generated for agent {agent_id}")
        if return_all:
            return component_outputs
        # Filter to only return actor/actor_critic outputs as final result
        actor_outputs = {}
        for component_id, output in component_outputs.items():
            component = self.components[component_id]
            if component.network_class in ["actor", "actor_critic"]:
                actor_outputs[component_id] = output
        
        return actor_outputs
    
    def evaluate(
        self, 
        obs: Union[Dict[str, torch.Tensor], torch.Tensor], 
        agent_id: Optional[str] = None
        ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value estimates for critic-based components based on their observations.
        
        Args:
            obs: Dictionary mapping agent IDs to their observations
            agent_id: Optional specific agent ID to evaluate. If None, evaluates all agents.
            
        Returns:
            Specific agent value estimates or all agent value estimates dictionary 
            mapping agent IDs to their value estimates
        """
        values_dict = {}
        
        if agent_id is not None:
            # Process only the specified agent and its dependencies
            if agent_id not in self.components:
                raise ValueError(f"Agent {agent_id} not found in components")
            
            # Get execution order and dependencies for the specific agent
            execution_order = self._determine_execution_order()
            agent_dependencies = self._get_agent_dependencies(agent_id, execution_order)
            
            # Process components in dependency order
            for component_id in execution_order:
                if self._is_required_for_agent(component_id, agent_id):
                    component = self.components[component_id]
                    
                    # Only evaluate critic-based components
                    if component.network_class in ["actor_critic", "critic"]:
                        # Use the observation for the target agent
                        if component_id == agent_id:
                            return component.evaluate(obs)
                        elif component_id in obs:
                            return component.evaluate(obs[component_id])
        else:
            if type(obs) != dict:
                raise ValueError(f"No specific observations provided.")
            # Original behavior: process all agents
            for agent_id, agent_obs in obs.items():
                if agent_id not in self.components:
                    raise ValueError(f"Agent {agent_id} not found in components")
                
                component = self.components[agent_id]
                if component.network_class in ["actor_critic", "critic"]:
                    values = component.evaluate(agent_obs)
                    values_dict[agent_id] = values
        
        if len(values_dict) == 0:
            target_info = f" for agent {agent_id}" if agent_id else " for any components"
            print(f"No values found{target_info}")
            
        return values_dict

    def get_actions_log_prob(
        self,
        actions: Union[Dict[str, torch.Tensor], torch.Tensor],
        agent_id: Optional[str] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the log probability of actions for all actor/actor-critic networks
        
        Args:
            actions: Dictionary mapping component IDs to their actions
            agent_id: Optional specific agent ID to get actions log probability. If None, gets all agents.
            
        Returns:
            Specific agent action log probabilities or all agent action log probabilities dictionary 
            mapping component IDs to their action log probabilities
            
        """
        log_probs = {}
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
            component = self.components[agent_id]
            if component.network_class in ["actor", "actor_critic"]:
                return component.get_actions_log_prob(actions)
        else:
            for component_id, component in self.components.items():
                if component.network_class in ["actor", "actor_critic"]:
                    if type(actions) != dict:
                        raise ValueError(f"No specific actions provided.")
                    if component_id not in actions:
                        raise ValueError(f"Actions for component {component_id} not found in actions dictionary")
                    log_probs[component_id] = component.get_actions_log_prob(actions[component_id])
                
        return log_probs
    
    def get_action_mean(
        self,
        agent_id: Optional[str] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the mean of the action distribution for all actor/actor-critic networks
        
        Args:
            agent_id: Optional specific agent ID to get action mean. If None, gets all agents.
            
        Returns:
            Specific agent action means or all agent action means dictionary 
            mapping component IDs to their action means
        """
        means = {}
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
            component = self.components[agent_id]
            if component.network_class in ["actor", "actor_critic"]:
                return component.get_action_mean()
        else:
            for component_id, component in self.components.items():
                if component.network_class in ["actor", "actor_critic"]:
                    means[component_id] = component.get_action_mean()
            return means

    def get_action_std(
        self,
        agent_id: Optional[str] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the standard deviation of the action distribution for all actor/actor-critic networks
        
        Args:
            agent_id: Optional specific agent ID to get action std. If None, gets all agents.
            
        Returns:
            Specific agent action standard deviations or all agent action standard deviations dictionary 
            mapping component IDs to their action standard deviations
        """
        stds = {}
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
            component = self.components[agent_id]
            if component.network_class in ["actor", "actor_critic"]:
                return component.get_action_std()
        else:
            for component_id, component in self.components.items():
                if component.network_class in ["actor", "actor_critic"]:
                    stds[component_id] = component.get_action_std()
            return stds
    
    def get_entropy(
        self,
        agent_id: Optional[str] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the entropy of the action distribution for all actor/actor-critic networks
        
        Args:
            agent_id: Optional specific agent ID to get entropy. If None, gets all agents.
            
        Returns:
            Specific agent entropy or all agent entropy dictionary 
            mapping component IDs to their entropy
        """
        entropies = {}
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
            component = self.components[agent_id]
            if component.network_class in ["actor", "actor_critic"]:
                return component.get_entropy()
        else:
            for component_id, component in self.components.items():
                if component.network_class in ["actor", "actor_critic"]:
                    entropies[component_id] = component.get_entropy()
            return entropies

                        
    def parameters(
        self, 
        agent_id: Optional[str] = None
        ) -> Union[Dict[str, Dict[str, List[torch.Tensor]]], Dict[str, List[torch.Tensor]]]:
        """Get all parameters of the policy.
        
        Args:
            agent_id: Optional specific agent ID to get parameters. If None, gets all agents.
            
        Returns:
            Specific agent parameters or all agent parameters dictionary 
            mapping component IDs to their parameters
        """
        params = {}
        if agent_id is not None:
            if agent_id not in self.components:
                raise ValueError(f"Agent ID {agent_id} not found in components")
            component = self.components[agent_id]
            return component.parameters()
        else:
            for component_id, component in self.components.items():
                params[component_id] = component.parameters()
        return params
    
    def save(self, path: str, component_ids: Optional[List[str]] = None):
        """Save all policies and critics to disk."""
        os.makedirs(path, exist_ok=True)
        
        if component_ids is not None:
            for component_id in component_ids:
                if component_id not in self.components:
                    raise ValueError(f"Component ID {component_id} not found in components")
                component = self.components[component_id]
                component_path = os.path.join(path, f"{component_id}.pt")
                component.save(component_path)
                print(f"Saved component '{component_id}' to {component_path}")
            return
        else:
            for component_id, component in self.components.items():
                component_path = os.path.join(path, f"{component_id}.pt")
                component.save(component_path)
        print(f"Saved all components to {path}")

    def load(self, path: str, component_ids: Optional[List[str]] = None):
        """
        Load a specific component from disk.
        
        Args:
            path: Path to the directory containing saved components
            component_ids: IDs of the components to load. If None, loads all components.
        """
        if component_ids is not None:
            # Load only the specified component
            for component_id in component_ids:
                if component_id in self.components:
                    component = self.components[component_id]
                    component_path = os.path.join(path, f"{component_id}.pt")
                    component.load(component_path)
                    print(f"Loaded component '{component_id}' from {component_path}")
                else:
                    raise ValueError(f"Component '{component_id}' not found in self.components")
        else:
            # Existing functionality to load all components (with bug fix)
            for comp_id, component in self.components.items():
                component_path = os.path.join(path, f"{comp_id}.pt")
                if not os.path.exists(component_path):
                    raise FileNotFoundError(f"Component file not found at {component_path}")
                component.load(component_path)
            print(f"Loaded all components from {path}")

    def _determine_execution_order(self):
        """Determine component execution order based on dependencies
        
        Raises:
            ValueError: If cyclic dependency detected
        """
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Cyclic dependency detected for component '{node}'")
            if node in visited:
                return
                
            temp_visited.add(node)
            
            # Visit dependencies
            if node in self.connections:
                for conn in self.connections[node]:
                    source_ids = conn['source_id']
                    for source_id in source_ids:
                        visit(source_id)
                    
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            
        # Visit all components
        for component in self.components:
            if component not in visited:
                visit(component)
                
        # Reverse to get correct order
        return list((order))
    
    def _validate_policy_structure(self):
        """Validate the policy structure
        
        Raises:
            ValueError: If component not found in components
            ValueError: If source not found in component outputs
        """
                
        # Validate connections
        for target, sources in self.connections.items():
            if target not in self.components:
                raise ValueError(f"Connection target '{target}' not found in components")
            for conn in sources:
                for source_id in conn['source_id']:
                    if source_id not in self.components:
                        raise ValueError(f"Connection source '{source_id}' not found in components")

    def _get_agent_dependencies(self, agent_id: str, execution_order: List[str]) -> List[str]:
        """Get the execution order including dependencies for a specific agent"""
        dependencies = set()
        to_process = [agent_id]
        
        while to_process:
            current = to_process.pop(0)
            if current in dependencies:
                continue
            dependencies.add(current)
            
            # Find components that this component depends on
            if current in self.connections:
                for conn in self.connections[current]:
                    for source_id in conn['source_id']:
                        if source_id not in dependencies:
                            to_process.append(source_id)
        
        # Return dependencies in original execution order
        return [comp_id for comp_id in execution_order if comp_id in dependencies]
    
    def _is_required_for_agent(self, component_id: str, target_agent_id: str) -> bool:
        """Check if a component is required for processing the target agent"""
        if component_id == target_agent_id:
            return True
        
        # Check if this component is a dependency of the target agent
        dependencies = self._get_agent_dependencies(target_agent_id, list(self.components.keys()))
        return component_id in dependencies