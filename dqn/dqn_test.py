# imports
import torch

# imports for environment
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

# importing the Deep Q-Network class
from dqn.dqn_training import DeepQNetwork

# main script
def test_dqn(weights_file):
    ale = ALEInterface()
    ale.loadROM(DoubleDunk)
    
    # initializing the environment
    env = gym.make('ALE/DoubleDunk-ram-v5', obs_type="ram", render_mode="human")

    # initializing the Deep Q-Network with 128 RAM input & 18 possible actions
    policy_net = DeepQNetwork(128, 18)

    # loading the pretrained weights
    policy_net.load_state_dict(torch.load(weights_file, weights_only=True))
    policy_net.eval()

    next_state, _ = env.reset()
    done = False

    # mapping indices of possible actions to readable label actions
    moves = {1:"NOOP", 2:"FIRE", 3:"UP", 4:"RIGHT", 5:"LEFT", 6:"DOWN", 7:"UPRIGHT", 8:"UPLEFT", 9:"DOWNRIGHT", 10:"DOWNLEFT", 11:"UPFIRE", 12:"RIGHTFIRE", 13:"LEFTFIRE", 14:"DOWNFIRE", 15:"UPRIGHTFIRE", 16:"UPLEFTFIRE", 17:"DOWNRIGHTFIRE", 18:"DOWNLEFTFIRE"}

    # execute an initial step using NOOP
    next_state, reward, done, truncated, info = env.step(1)
    
    # loop until the game is done
    while not done:
        policy_net.eval()
        with torch.no_grad():
            # compute action probabilities from the current state of the game
            action_probabilities = policy_net(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
            action_probabilities = torch.softmax(action_probabilities, dim=1)
            action = torch.multinomial(action_probabilities, 1).item()
        next_state, reward, done, truncated, info = env.step(action)
        # next_state, reward, done, truncated, info = env.step(action_probabilities.argmax().item())
        env.render()
    input()
    env.close()

