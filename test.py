# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:30:42 2021

@author: User
"""

import gym
import bongo_board # 
import time
env = gym.make('BongoBoard-v0')
for i_episode in range(20):
    env.reset()
    for t in range(100):
        action = env.action_space.sample()
        # print(action)
        env.step(action)
        env.render()
        time.sleep(0.01)
        # print(env.test())
        # env.render()
        # print(observation)
        # action = env.action_space.sample()
        # observation, reward, done, info = env.step(action)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
env.close()