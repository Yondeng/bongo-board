import gym
import math
import random
import numpy as np
from DQN.DQN_Model import *
import bongo_board
from matplotlib import pyplot as plt

env = gym.make('BongoBoard-v0')

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Hyper parameters
n_hidden = 100
batch_size = 128
lr = 0.0001                 
epsilon = 0.1             
gamma = 0.9             
target_replace_iter = 100 
memory_capacity = 4000
n_episodes = 500

# Build DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)
rewardsArr = []
def main():
    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        state = env.reset()
        while True:
            if i_episode > 300:
                env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            alpha, alpha_dot, theta, theta_dot = next_state
            r1 = (math.pi/2 - abs(alpha))
            r2 = (env.max_theta - abs(theta))
            if done:
                reward = 0
            else:
                reward = r1 + r2
            dqn.store_transition(state, action, reward, next_state)

            rewards += reward
            if dqn.memory_counter > memory_capacity:
                dqn.learn()
            state = next_state
            if done:
                print('Episode finished after {}\
                    timesteps, total rewards {} Epoch {}'.format(t+1, rewards, i_episode))
                rewardsArr.append(rewards)
                break
            t += 1
    env.close()
def plot_Reward(self):
    plt.plot(rewardsArr)
    plt.show()
if __name__ == "__main__":
    main()
    plot_Reward()