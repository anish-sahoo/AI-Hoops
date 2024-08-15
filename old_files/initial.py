import torch
import numpy as np
import polars as pd
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

moves = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

obs = env.reset()
for i in range(500):
    action = env.action_space.sample()
    print(moves[action])
    obs, reward, truncated, info, done = env.step(action)
    # print(reward, done, info)
    time.sleep(0.1)

env.close()