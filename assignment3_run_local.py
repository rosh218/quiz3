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

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

TRAIN = False  # if set to false will skip training, load the last saved model and use that for testing

# Hyper parameters that will be used in the DQN algorithm

EPISODES = 2500  # number of episodes to run the training for
LEARNING_RATE = 0.0005  # the learning rate for optimising the neural network weights
MEM_SIZE = 50000  # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 10000  # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 64  # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99  # Discount factor
EPS_START = 0.1  # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001  # Final epsilon value
EPS_DECAY = 6 * MEM_SIZE  # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1  # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 5000  # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 256  # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 256  # Number of neurons in our MLP's second hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_
np.bool8 = np.bool_

# Path to trained model
MODEL_PATH = os.path.join("Models", "policy_network.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),  # input layer
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),  # hidden layer
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(FC2_DIMS, self.action_space),  # output layer
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.HuberLoss()

    def forward(self, x):
        return self.layers(x)

# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            ############ avoid catastropic forgetting - retain initial 10% of the replay buffer ##############
            mem_index = int(self.mem_count % ((1 - MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))
            ##################################################################################################

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0  # keep track of the number of iterations we have learnt for

    # epsilon greedy
    def choose_action(self, observation):
        # only start decaying epsilon once we actually start learning, i.e. once the replay memory has REPLAY_START_SIZE
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0

        # if we rolled a value lower than epsilon sample a random action
        if random.random() < eps_threshold:
            action_probabilities = np.array([0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1])
            action_probabilities /= action_probabilities.sum()
            return np.random.choice(self.policy_network.action_space, p=action_probabilities)
        else:
            state = torch.tensor(observation).float().unsqueeze(0)
            self.policy_network.eval()  # only need forward pass
            with torch.no_grad():  # so we don't compute gradients - save memory and computation
                ################ retrieve q-values from policy network, Q ################################
                q_values = self.policy_network(state)
                ##########################################################################################
            return torch.argmax(q_values).item()

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = (
            self.memory.sample())  # retrieve random batch of samples from replay memory
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)[batch_indices, actions]

        self.target_network.eval()  # only need forward pass
        with torch.no_grad():  # so we don't compute gradients - save memory and computation
            ###### get q-values of states_ from target network, \hat{q}, for computation of the target q-values ###############
            q_values_next = self.target_network(states_)
            ###################################################################################################################

        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # max q values for next state
        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value

        ###### compute loss between target (from target network, \hat{Q}) and estimated q-values (from policy network, Q) #########
        loss = self.policy_network.loss(q_values, q_target)
        ###########################################################################################################################

        #compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

# Train network
if TRAIN:
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    episode_batch_score = 0
    episode_reward = 0
    agent = DQN_Solver(env)  # create DQN agent
    plt.clf()

    for i in range(EPISODES):
        state, info = env.reset()
        while True:
            action = agent.choose_action(state)
            state_, reward, done, _, info = env.step(action)

            if math.sqrt(state_[0] ** 2 + state_[1] ** 2) < 1.5:
                reward += 50

            agent.memory.add(state, action, reward, state_, done)

            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()

            state = state_
            episode_batch_score += reward
            episode_reward += reward

            if done:
                break

        episode_history.append(i)
        episode_reward_history.append(episode_reward)

        if len(episode_reward_history) >= 150:
            last_rewards = episode_reward_history[-150:]
            average_reward = np.mean(episode_reward_history)
            std_deviation = np.std(last_rewards)

            if episode_reward > (average_reward + 3 * std_deviation):
                print("Converging, Current reward: ", round(episode_reward, 2), "Average Reward: ", round(average_reward, 2))

        episode_reward = 0.0

        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            torch.save(agent.policy_network.state_dict(), MODEL_PATH)
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score / float(100))
            print("epsilon: ", EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * agent.learn_count / EPS_DECAY))
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            episode_batch_score = 0

# Test trained policy
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

agent = DQN_Solver(env)
agent.policy_network.load_state_dict(torch.load(MODEL_PATH))
agent.policy_network.eval()

state, info = env.reset()

while True:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)

    action = torch.argmax(q_values).item()
    state_, reward, done, _, info = env.step(action)

    if math.sqrt(state_[0] ** 2 + state_[1] ** 2) < 1.5:
        reward += 50

    print(f"Reward: {reward:.2f} | Distance to goal: {math.sqrt(state_[0] ** 2 + state_[1] ** 2):.2f}")

    state = state_

    if done:
        print("done: " + str(done))
        break

env.close()
