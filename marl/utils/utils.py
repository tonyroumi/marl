import torch
import numpy as np
import random

from robosuite.controllers import load_composite_controller_config

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from a string.
    
    Raises:
        ValueError: [Invalid activation function]
    """
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")
    
def resolve_controller(controller_config):
    """Resolve the controller from a configuration dictionary.
    
    Args:
        controller_config (dict): Controller configuration dictionary
            should contain at least 'type' key
    
    Returns:
        Controller object based on the configuration
        
    Raises:
        ValueError: If controller type is invalid or misconfigured
    """
    controller_type = controller_config.get('type')
    
    if controller_type == "composite_controller":
        controller_name = controller_config.get('controller', "OSC_POSE")
        return load_composite_controller_config(controller_name)
    
    else:
        raise ValueError(f"Invalid controller type '{controller_type}'.")

def set_seed(seed: int):
    """ Set seed for reproducibility """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)   