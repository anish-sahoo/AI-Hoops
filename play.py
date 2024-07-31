# could use python 3.11
import numpy as np
import pandas as pd

import multiprocessing as mp
import json
import shutil as sl
import os

import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

ale = ALEInterface()
ale.loadROM(DoubleDunk)

env = gym.make('ALE/DoubleDunk-v5', render_mode='human', obs_type="ram")


def play(pair):
    dir, num = pair
    obs = env.reset()
    total_reward = 0
    data = []
    terminated = False
    truncated = False
    while not (terminated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        new_obs = np.array([obs])
        new_rew = np.array([reward])
        data.append((new_obs.tolist(), new_rew.tolist()))

    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + "/data" + str(num) + ".json"
    f = open(path, "a")
    f.write(json.dumps(data))
    f.close()

    env.close()

def run(num_instances, num_total, dir):
    with mp.Pool(num_instances) as p:
        p.map(play, zip([dir]*num_total, range(num_total)))
    sl.make_archive(dir+"_compressed", "zip", dir)
    sl.rmtree(dir)

if __name__ == "__main__":
    num_instances = 10
    runs_per_batch = 1000
    
    i = 0
    while True:
        dir = "data"
        while os.path.exists(dir + str(i)) or os.path.isfile(dir + str(i) + "_compressed.zip"):
            i += 1
        run(num_instances, runs_per_batch, dir + str(i))
        i += 1