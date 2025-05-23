from marl.policies.component import Component
from marl.policies.multi_agent_policy import MultiAgentPolicy
from typing import List

class MultiAgentPolicyBuilder:
    """
    A builder pattern implementation for constructing complex multi-agent reinforcement learning policies.
    
    This class provides a fluent interface for incrementally building multi-agent systems by:
    - Adding individual neural network components (actors, critics, encoders)
    - Defining data flow connections between components
    - Validating component relationships before construction
    
    Example:
        >>> builder = MultiAgentPolicyBuilder()
        >>> policy = (builder
        ...     .add_component("encoder", "cnn", "encoder", input_shape=(84, 84, 4))
        ...     .add_component("actor", "mlp", "actor", hidden_dims=[256, 256])
        ...     .add_component("critic", "mlp", "critic", hidden_dims=[256, 256])
        ...     .add_connection(["encoder"], "actor") #encoder -> actor
        ...     .add_connection(["encoder"], "critic") #encoder -> critic
        ...     .build())
    
    Raises:
        ValueError: If attempting to add a connection with a non-existent source component.
        ValueError: If attempting to add a connection with a non-existent target component.
        ValueError: If invalid network parameters are provided to component creation.
    
    """
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
        self.components[component_id] = Component(
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
            