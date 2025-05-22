import torch
from typing import Dict, Any, List, Optional
from marl.policies import Policy
import os

class MultiAgentPolicy(Policy):
    """
    Multi-agent policy that can be configured for various sharing patterns.
    
    This class provides a uniform interface for different multi-agent policy configurations:
    - Shared policy: All agents share the same policy
    - Independent policies: Each agent has its own policy
    - Mixed: Some agents share policies, others have independent ones
    - Shared critic: Agents have independent actors but share a critic
    """
    def __init__(
        self, 
        components: Dict[str, Policy],
        connections: Dict[str, Dict[str, Any]],
        ):
        """
        Initialize multi-agent policy.
        
        Args:
            components: Dictionary of components in the policy
            connections: Dictionary of connections between components
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.components = components
        self.connections = connections

        #Validate the policy structure
        self._validate_policy_structure()
    
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

    def act(
        self, 
        obs: Dict[str, torch.Tensor], 
        deterministic: bool = False
        ) -> Dict[str, torch.Tensor]:
        """ Process observations through the policy components
        
        Args:
            obs: Dictionary mapping agent IDs to their observations
            deterministic: Whether to use deterministic actions
        
        Raises:
            ValueError: If component not found in components
            ValueError: If source not found in component outputs
            
        Returns:
            Dictionary mapping agent IDs to their actions
        """
        component_outputs = {}
        component_info = {}
        execution_order = self._determine_execution_order()
        for component_id in execution_order:
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not found in components")
            
            component = self.components[component_id]
            if component_id in self.connections:
                for conn in self.connections[component_id]:
                    source_ids = conn['source_id']
                    concat_dim = conn['concat_dim']
                    target_inputs = []
                    for source_id in source_ids:
                        if source_id not in component_outputs:
                            raise ValueError(f"Source {source_id} not found in component outputs")
                        if source_id not in component_outputs:
                            raise ValueError(f"Source {source_id} not found in component outputs")
                        target_inputs.append(component_outputs[source_id])
                target_inputs = torch.cat(target_inputs, dim=concat_dim) if len(target_inputs) > 1 else target_inputs[0]
                if component_id in obs:
                    target_inputs = torch.cat([target_inputs, obs[component_id]], dim=concat_dim)
                component_outputs[component_id] = component.forward(target_inputs)
            else:
                if component.network_class in ["actor", "actor_critic"]:
                    component_outputs[component_id], component_info[component_id] = component.act(obs[component_id], deterministic=deterministic)
                elif component.network_class == "encoder":
                    component_outputs[component_id] = component.forward(obs[component_id])
        return component_outputs, component_info
    
    def evaluate(
        self, 
        obs: Dict[str, torch.Tensor], 
        ) -> Dict[str, torch.Tensor]:
        """
        Get value estimates for all critc-based components based on their observations.
        
        Args:
            obs: Dictionary mapping agent IDs to their observations
            
        Returns:
            Dictionary mapping agent IDs to their value estimates
        """
        values_dict = {}

        for agent_id, agent_obs in obs.items():
            component = self.components[agent_id]
            if agent_id not in self.components:
                raise ValueError(f"Agent {agent_id} not found in components")
            if component.network_class in ["actor_critic", "critic"]:
                values = component.evaluate(agent_obs)
                values_dict[agent_id] = values
        if len(values_dict) == 0:
            print(f"No values found for for any components")
        return values_dict

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
                        
    def parameters(self) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Get all parameters of the policy."""
        params = {}
        for component_id, component in self.components.items():
            params[component_id] = component.parameters()
        return params
    
    def save(self, path: str):
        """Save all policies and critics to disk."""
        os.makedirs(path, exist_ok=True)
        
        for component_id, component in self.components.items():
            component_path = os.path.join(path, f"{component_id}.pt")
            component.save(component_path)
        print(f"Saved all components to {path}")

    def load(self, path: str, component_id: Optional[str] = None):
        """
        Load a specific component from disk.
        
        Args:
            path: Path to the directory containing saved components
            component_id: ID of the specific component to load. If None, loads all components.
        """
        if component_id is not None:
            # Load only the specified component
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
                component.load(component_path)
            print(f"Loaded all components from {path}")
