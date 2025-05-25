from marl.policies.component import Component
from marl.policies.multi_agent_policy import MultiAgentPolicy
from typing import List, Dict, Any
import torch

class MultiAgentPolicyBuilder:
    """
    A builder pattern implementation for constructing complex multi-agent reinforcement learning policies.
    
    This class provides a fluent interface for incrementally building multi-agent systems by:
    - Adding individual neural network components (actors, critics, encoders)
    - Defining data flow connections between components
    - Loading pre-trained components and freezing their weights
    - Validating component relationships before construction
    
    Example:
        >>> builder = MultiAgentPolicyBuilder()
        >>> policy = (builder
        ...     .add_component("encoder", "cnn", "encoder", input_shape=(84, 84, 4))
        ...     .load_component("encoder", "pretrained_encoder.pth")
        ...     .freeze_component("encoder")
        ...     .add_component("actor", "mlp", "actor", hidden_dims=[256, 256])
        ...     .add_component("critic", "mlp", "critic", hidden_dims=[256, 256])
        ...     .add_connection(["encoder"], "actor")
        ...     .add_connection(["encoder"], "critic")
        ...     .build())
    
    Raises:
        ValueError: If attempting to add a connection with a non-existent source component.
        ValueError: If attempting to add a connection with a non-existent target component.
        ValueError: If invalid network parameters are provided to component creation.
        ValueError: If attempting to freeze a non-existent component.
    
    """
    def __init__(self):
        self.components = {}
        self.connections = {}
        self.frozen_components = set()

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

    def load_component(self, component_id: str, path: str):
        """
        Load pre-trained weights for a component.
        
        Args:
            component_id: The ID of the component to load
            path: Path to the saved model weights
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")
        
        self.components[component_id].load(path)
        print(f"Loaded weights for component {component_id} from {path}")
        return self

    def freeze_component(self, component_id: str):
        """
        Freeze all parameters in a component.
        
        Args:
            component_id: The ID of the component to freeze
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")
        
        self.components[component_id].freeze()
        self.frozen_components.add(component_id)
        return self

    def unfreeze_component(self, component_id: str):
        """
        Unfreeze all parameters in a component.
        
        Args:
            component_id: The ID of the component to unfreeze
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")
        
        self.components[component_id].unfreeze()
        self.frozen_components.discard(component_id)
        return self

    def freeze_component_layers(self, component_id: str, layer_names: List[str]):
        """
        Freeze specific layers in a component.
        
        Args:
            component_id: The ID of the component
            layer_names: List of layer names to freeze
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")
        
        self.components[component_id].freeze_layers(layer_names)
        return self

    def freeze_components(self, component_ids: List[str]):
        """
        Freeze multiple components at once.
        
        Args:
            component_ids: List of component IDs to freeze
        """
        for component_id in component_ids:
            self.freeze_component(component_id)
        return self

    def load_and_freeze_component(self, component_id: str, path: str):
        """
        Load pre-trained weights and immediately freeze the component.
        
        Args:
            component_id: The ID of the component
            path: Path to the saved model weights
        """
        return self.load_component(component_id, path).freeze_component(component_id)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get all trainable parameters from all components.
        
        Returns:
            List of trainable parameters across all components
        """
        trainable_params = []
        for component in self.components.values():
            trainable_params.extend(component.get_trainable_parameters())
        return trainable_params

    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Get a summary of parameters across all components.
        
        Returns:
            Dictionary with parameter statistics for each component
        """
        summary = {}
        total_params = 0
        total_trainable = 0
        
        for component_id, component in self.components.items():
            info = component.get_parameter_info()
            summary[component_id] = info
            total_params += info['total_parameters']
            total_trainable += info['trainable_parameters']
        
        summary['overall'] = {
            'total_parameters': total_params,
            'trainable_parameters': total_trainable,
            'frozen_parameters': total_params - total_trainable,
            'frozen_percentage': (total_params - total_trainable) / total_params * 100 if total_params > 0 else 0,
            'frozen_components': list(self.frozen_components)
        }
        
        return summary

    def print_parameter_summary(self):
        """Print a formatted parameter summary."""
        summary = self.get_parameter_summary()
        
        print("\n" + "="*60)
        print("PARAMETER SUMMARY")
        print("="*60)
        
        for component_id, info in summary.items():
            if component_id == 'overall':
                continue
            
            frozen_status = "FROZEN" if info['is_frozen'] else "TRAINABLE"
            print(f"\n{component_id.upper()} ({info['network_class']}) - {frozen_status}")
            print(f"  Total params: {info['total_parameters']:,}")
            print(f"  Trainable:    {info['trainable_parameters']:,}")
            print(f"  Frozen:       {info['frozen_parameters']:,} ({info['frozen_percentage']:.1f}%)")
        
        overall = summary['overall']
        print(f"\nOVERALL SUMMARY")
        print(f"  Total params: {overall['total_parameters']:,}")
        print(f"  Trainable:    {overall['trainable_parameters']:,}")
        print(f"  Frozen:       {overall['frozen_parameters']:,} ({overall['frozen_percentage']:.1f}%)")
        print(f"  Frozen components: {', '.join(overall['frozen_components']) if overall['frozen_components'] else 'None'}")
        print("="*60)

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
        """
        Build the multi-agent policy.
        
        Returns:
            A MultiAgentPolicy instance
        """
        # Print parameter summary before building
        self.print_parameter_summary()
        
        return MultiAgentPolicy(
            components=self.components,
            connections=self.connections,
        )