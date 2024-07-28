# could use python 3.11
import torch
import keras
import numpy as np
import pandas as pd
import json

import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human', obs_type="ram")

f = open("meta_data.txt", "r")
size = json.loads(f.read())["ram_size"]
f.close()


# define the keras model
model = Sequential()
model.add(Dense(env.observation_space.shape[0]*2, input_shape=env.observation_space.shape, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

f = open("data1.txt", "r")
data = json.loads(str(f.read()))
f.close()

model.fit([x[0] for x in data], [x[1] for x in data])
_, accuracy = model.evaluate([x[0] for x in data], [x[1] for x in data], verbose=0)
print("accuracy: ", accuracy)
env.close()