# MountainCar-v0
The goal of the agent is to drive the car up the mountain and reach the goal at the top of the right hill as quickly as possible.

## Overview
This project demonstrates a reinforcement learning approach to solving the Mountain Car problem using Q-learning. The code is designed to train an agent to drive a car up a steep hill using discrete actions.

## Description
The Mountain Car problem is a classic reinforcement learning task where an underpowered car must drive up a steep hill. The car must leverage momentum to reach the goal at the top of the hill.

## Components
Environment Initialization: The simulation uses the MountainCar-v0 environment from the Gymnasium library. This environment provides the car's position and velocity as observations and allows for three actions: drive left, stay neutral, and drive right.

State Discretization: Continuous state space (position and velocity) is discretized into bins to create a manageable state space for the Q-learning algorithm.

## Q-Table Initialization:

In training mode, the Q-table is initialized with random values.
In evaluation mode, a pre-trained Q-table is loaded from a file.
Training Parameters:

Alpha (α): Learning rate, controls the update step size.
Gamma (γ): Discount factor, determines the importance of future rewards.
Epsilon (ε): Epsilon-greedy policy parameter, balances exploration and exploitation. It decays over time to reduce exploration as the agent learns.
Training Loop: The agent is trained over a specified number of episodes:

The agent's state is reset at the beginning of each episode.
For each step within an episode, the agent selects an action based on the ε-greedy policy.
The environment responds with a new state and reward.
The Q-table is updated using the Q-learning formula.
Epsilon is decayed to reduce exploration over time.
Rewards are logged for performance tracking.
Reward Shaping: An optional feature to modify the reward structure to speed up learning by providing intermediate rewards.

## Saving and Plotting:

The trained Q-table is saved to a file after training.
The average rewards over episodes are plotted to visualize training progress.
The plot and reward data are saved to files for further analysis.

## Requirements
* Python 3.x
* Gymnasium library
* NumPy library
* Matplotlib library
* Installation
* Install the required libraries using pip:

