# train_q_learning.py
import gym
import simple_driving
import numpy as np
import pickle
import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Discretization setup
NUM_BINS = 20
X_BINS = np.linspace(-40, 40, NUM_BINS)
Y_BINS = np.linspace(-40, 40, NUM_BINS)

# Path to save model
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Model'))
os.makedirs(MODEL_DIR, exist_ok=True)

def discretize_state(state):
    x, y = state
    x_bin = np.digitize(x, X_BINS)
    y_bin = np.digitize(y, Y_BINS)
    return (x_bin, y_bin)

def epsilon_greedy(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(np.arange(9), p=[0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.15, 0.4, 0.1])
    return np.argmax(q_table[state])

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 1000

# Initialize environment (headless for faster training)
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

returns = []

for ep in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        action = epsilon_greedy(q_table, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        if reward > -1.5:
            reward += 50  # Goal bonus

        q_old = q_table[state][action]
        q_next_max = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * q_next_max - q_old)

        state = next_state
        total_reward += reward

    returns.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

# Save Q-table
model_path = os.path.join(MODEL_DIR, "q_table.pkl")
with open(model_path, "wb") as f:
    pickle.dump(dict(q_table), f)

# Plot training curve
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Training Progress")
plt.grid()
plt.show()

env.close()
