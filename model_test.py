import numpy as np
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk
import matplotlib.pyplot as plt
from doubledunk_training import DeepQNetwork, get_action_probabilities
import torch

ale = ALEInterface()
ale.loadROM(DoubleDunk)
env = gym.make('ALE/DoubleDunk-ram-v5', obs_type="ram", render_mode="human")

policy_net = DeepQNetwork(128, 18)

policy_net.load_state_dict(torch.load('policy_net.pth', weights_only=True))
policy_net.eval()

next_state, _ = env.reset()
done = False

moves = {1:"NOOP", 2:"FIRE", 3:"UP", 4:"RIGHT", 5:"LEFT", 6:"DOWN", 7:"UPRIGHT", 8:"UPLEFT", 9:"DOWNRIGHT", 10:"DOWNLEFT", 11:"UPFIRE", 12:"RIGHTFIRE", 13:"LEFTFIRE", 14:"DOWNFIRE", 15:"UPRIGHTFIRE", 16:"UPLEFTFIRE", 17:"DOWNRIGHTFIRE", 18:"DOWNLEFTFIRE"}

while not done:
    action_probabilities = get_action_probabilities(policy_net, next_state)
    print(moves[action_probabilities.argmax().item()])
    next_state, reward, done, truncated, info = env.step(action_probabilities.argmax().item())
    env.render()
    
env.close()