import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import os

# Load trained Q-table
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'q_table.pkl'))
with open(model_path, "rb") as f:
    q_table = pickle.load(f)

# Discretization bins
NUM_BINS = 20
X_BINS = np.linspace(-40, 40, NUM_BINS)
Y_BINS = np.linspace(-40, 40, NUM_BINS)

def discretize_state(state):
    x, y = state
    x_bin = np.digitize(x, X_BINS)
    y_bin = np.digitize(y, Y_BINS)
    return (x_bin, y_bin)

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped
##########################################################################################################################

state, info = env.reset()
state = discretize_state(state)
# frames = []
# frames.append(env.render())

for i in range(200):
    action = np.argmax(q_table.get(state, np.zeros(9)))
    next_state, reward, done, _, info = env.step(action)
    state = discretize_state(next_state)
    state, reward, done, _, info = env.step(action)
    # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    if done:
        break

env.close()
