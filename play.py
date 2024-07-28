# could use python 3.11
import numpy as np
import pandas as pd

import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human', obs_type="ram")

import json

obs = env.reset()
total_reward = 0
data = []
terminated = False
truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    new_obs = np.array([obs])
    new_rew = np.array([reward])
    data.append((new_obs.tolist(), new_rew.tolist()))

f = open("data1.txt", "a")
f.write(json.dumps(data))
f.close()

env.close()