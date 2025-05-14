from marl.policies.policy import Policy
from marl.policies.multi_agent_policy import MultiAgentPolicy
from typing import Dict, List

class MultiAgentPolicyBuilder:
    """ Builder for multi-agent policies. """
    def __init__(self):
        self.components = {}
        self.connections = {}

    def add_component(
        self, 
        component_id: str, 
        network_type: str,
        network_class: str,
        **network_kwargs
        ):
        """
        Add a component to the multi-agent policy.
        
        Args:
            component_id: str,
            network_type: str,
            network_class: str,
            **network_kwargs
        """
        self.components[component_id] = Policy(
            component_id=component_id,
            network_type=network_type,
            network_class=network_class,
            **network_kwargs
        )
        return self

    def add_connection(
        self, 
        source_id: List[str], 
        target_id: str,
        concat_dim: int = -1,
        ):
        """
        Add a connection between two components.
        
        Args:
            source_id: The source component ID (where output is coming from)
            target_id: The target component ID (where output is going to)
            concat_dim: The dimension to concatenate on
        """
        for source in source_id:
            if source not in self.components:
                raise ValueError(f"Source component {source} not found")
        if target_id not in self.components:
            raise ValueError(f"Target component {target_id} not found")
        
        if target_id not in self.connections:
            self.connections[target_id] = []
        self.connections[target_id].append({
            "source_id": source_id,
            "concat_dim": concat_dim
        })
        return self
    
    
    def build(self):
        """Build the multi-agent policy.
        
        Returns:
            A MultiAgentPolicy instance
        """
        return MultiAgentPolicy(
            components=self.components,
            connections=self.connections,
        )
            