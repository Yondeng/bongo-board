# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:30:42 2021

@author: User
"""

import gym
import bongo_board # 
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('BongoBoard-v0')
for i_episode in range(20):
    env.reset()
    while True:
        action = env.action_space.sample()
        # print(action)
        state, reward, done,_ = env.step(action)
        img = env.render(mode='rgb_array').transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        print(np.shape(img))
        plt.imshow(img.permute(1, 2, 0).numpy(),
           interpolation='none')
        plt.title('Example extracted screen')
        plt.show()
        print(img)
        break
        # env.ale.saveScreenPNG('test_image2.png')
        # print(done)
        if done == True:
            break
        time.sleep(0.01)
        # print(env.test())
        # env.render()
        # print(observation)
        # action = env.action_space.sample()
        # observation, reward, done, info = env.step(action)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
    break
env.close()



