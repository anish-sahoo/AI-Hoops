import torch
import keras
import numpy as np
import pandas as pd
import random
import time

import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human', obs_type="ram")

#0s env.observation_space.low
#255 env.observation_space.high
# (210, 160, 3) env.observation_space.shape
# uint8 env.observation_space.dtype
# method env.observation_space.seed
# -inf to #inf env.reward_range

# define the keras model
model = Sequential()
model.add(Dense(env.observation_space.shape[0]*2, input_shape=env.observation_space.shape, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

obs = env.reset()
total_reward = 0
data = []
for i in range(5000):
    action = env.action_space.sample()
    obs, reward, truncated, info, done = env.step(action)
    total_reward += reward

    new_obs = np.array([obs])
    new_rew = np.array([reward])
    data.append((new_obs, new_rew))
    model.fit(new_obs, new_rew)
    if not done:
        break


_, accuracy = model.evaluate([x[0] for x in data], [x[1] for x in data], verbose=0)
print("accuracy: ", accuracy)
env.close()