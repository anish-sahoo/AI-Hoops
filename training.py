# could use python 3.11
import torch
import keras
import numpy as np
import pandas as pd

import json
import zipfile as zf
import io
import os


import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human', obs_type="ram")

def generate_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(env.observation_space.shape[0]*2, input_shape=env.observation_space.shape, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_data():
    dir = "data"
    max_i = 0
    while os.path.isfile(dir + str(max_i) + "_compressed.zip"):
        max_i += 1

    for i in range(max_i):
        dir = "data" + str(i)
        zip_name = dir + "_compressed.zip"

        with zf.ZipFile(zip_name) as f:
            file_names = f.namelist()
            for file_name in file_names:
                with io.TextIOWrapper(f.open(file_name), encoding="utf-8") as data_file:
                    yield json.loads(str(data_file.read()))

def train_model(model):
    for data in generate_data():
        print(len(data))
        for data_set in data:
            if(np.array(data_set[0][0]).shape != 128):
                print(np.array(data_set[0][0]).shape)
            if(np.array(data_set[1]).shape != 1):
                print(np.array(data_set[1]).shape)
            model.fit(np.array(data_set[0][0]), data_set[1])

def main():
    model = generate_model()
    train_model(model)

if __name__ == "__main__":
    main()

'''
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
'''