import torch
import numpy as np
import pandas as pd
import random
import time

import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human')

print(env.observation_space)
print(str(env.action_space))
print(env.reward_range)
print(env.metadata)
print(env.spec)
print(env.unwrapped.get_action_meanings())

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, truncated, info, done = env.step(action)
    # print(reward, done, info)
    time.sleep(0.01)

env.close()