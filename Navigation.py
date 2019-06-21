#!/usr/bin/env python
# coding: utf-8

from unityagents import UnityEnvironment
import torch
import numpy as np

#initialize Environment
#env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
env = UnityEnvironment(file_name="/home/user/data_github/Udacity/ReinforcementLearning/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
