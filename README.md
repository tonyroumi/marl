# Multi-Agent Reinforcement Learning Framework

A modular, configuration-driven multi-agent reinforcement learning framework built with Hydra for easy experimentation and component swapping.

## Overview

This codebase provides a flexible foundation for multi-agent reinforcement learning research and experimentation. The architecture emphasizes modularity and configurability, allowing researchers to easily swap algorithms, environments, policies, and network architectures through simple configuration changes.

### Key Features

- **Configuration-driven architecture** using Hydra for seamless component swapping
- **Multi-agent support** with flexible policy composition and inter-agent communication
- **Multiple RL algorithms** including PPO and MAPPO
- **Modular network architectures** with factory patterns for easy extension
- **Efficient data collection** with rollout buffers and GAE advantage computation
- **Environment wrappers** for standardized interfaces across different environments



## Quick Start

### 1. Environment Setup

First, ensure you have the required dependencies installed. The codebase uses Hydra for configuration management and supports various RL environments.

### 2. Configuration

The main configuration is managed through `.configs/config.yaml`. This file orchestrates all components:

```yaml
# Example configuration structure
agent: marl_agent
algorithm: mappo
environment: your_env
policy: multi_agent
```

All components can be easily swapped by modifying the configuration files in their respective directories.

### 3. Running Experiments

```python
# Basic training loop
python train.py
```

The framework will automatically load configurations and initialize all components according to your specifications.

### 4. Inspecting Environments

Before training, use the inspector tool to understand your environment's observation space:

```python
python tools/inspector.py
```

This will print detailed information about:
- Observation types and shapes
- Available observation modalities
- Image data specifications (if present)

## Core Components

### Agents

The agent system follows a standard on-policy RL pipeline:

- **Base Agent**: Foundation class providing common functionality
- **MARL Agent**: Multi-agent implementation with standard RL pipeline including:
  - `learn()` function for training iterations
  - Rollout collection from environments
  - Algorithm-based return computation
  - Policy updates

### Algorithms

- **PPO**: Supports multiple actor-critics and diverse policy variations
- **MAPPO**: Multi-agent extension of PPO for coordinated learning

### Policies

The policy system uses a composite architecture:

- **Policy Builder**: Creates policy containers from configurations
- **Multi-Agent Policy**: Orchestrates multiple interconnected components including:
  - Individual agent architectures with different network types
  - Inter-component communication through configurable connections
  - Selective agent processing for efficient inference

### Networks

Modular network architectures with factory patterns:

- **MLP Actor/Critic**: Standard multilayer perceptron implementations
- **Actor-Critic Networks**: Combined architectures for value-based methods
- **Network Factory**: Simplified agent creation and configuration

### Storage

Efficient data management for training:

- **Rollout Storage**: Manages trajectories from parallel environments with:
  - GAE (Generalized Advantage Estimation) computation
  - Mini-batch sampling for optimization
- **World Model Storage**: Dedicated buffer for world model training scenarios

## Configuration System

The Hydra-based configuration system allows for:

- **Easy experimentation** through configuration file modifications
- **Component isolation** enabling independent testing of algorithms, environments, and policies
- **Reproducible experiments** with version-controlled configurations
- **Hyperparameter sweeps** using Hydra's built-in capabilities

### Configuration Categories

- **Agent configs**: Define agent behavior and learning parameters
- **Algorithm configs**: Specify RL algorithm settings and hyperparameters
- **Environment configs**: Configure environment parameters and wrappers
- **Policy configs**: Define policy architectures and multi-agent coordination

## Usage Examples

### Basic Training

```python
# The framework handles initialization automatically
# Modify .configs/config.yaml to change components
```


### New Algorithms

1. Implement algorithm in `algorithms/`
2. Add configuration file in `.configs/algorithm/`
3. Update main config to use new algorithm


## Adding New Environments
This codebase supports multi-agent reinforcement learning (MARL) environments that follow the Gymnasium format. Environments are automatically vectorized using Gymnasium's `AsyncVectorEnv` and `SyncVectorEnv` for parallel execution.

### Environment Requirements
**Observation Format**: Since this is a MARL codebase, environments must return observations as dictionaries to handle multi-agent scenarios properly.

**Compatibility**: Environments must follow the standard Gymnasium interface and be compatible with vectorization.

### Supported Environment Types
The codebase currently supports:
- **Gymnasium environments**
- **Robosuite environments**

### Configuration Setup
To add a new environment, creatre a configuration file under [.configs/environments](marl/.configs/environment) 
- **id**: The environment identifier
- **type**: Environment execution type:
- **num_envs**: Number of parallel environments
- **seed**: Random seed for reporoducibility
- **env_kwargs**: Environment-specific parameters passed to the constructor 




