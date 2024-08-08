# %%
import polars as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import relu
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset, random_split
import random

# %%
torch.device("cuda")

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        state = np.array(state)
        next_state = np.array(next_state)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.buffer)


# %%
# class DeepQNetwork(nn.Module):
#     def __init__(self, input_dim, action_space):
#         super(DeepQNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 32)
#         self.fc5 = nn.Linear(32, action_space)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )
    
    def forward(self, x):
        return self.net(x)

# %%
def epsilon_greedy_action_selection(policy_net, state, epsilon):
    if np.random.rand() < epsilon:
        # return np.random.randint(policy_net.fc5.out_features)
        return np.random.randint(policy_net.net[-1].out_features)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to("cuda")
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.max(1)[1].item()

# %%
def train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq):
    input_dim = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    policy_net = DeepQNetwork(input_dim, action_space).to("cuda")
    target_net = DeepQNetwork(input_dim, action_space).to("cuda")
    
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    
    replay_buffer = ReplayBuffer(capacity=10000)
    
    epoch_losses = []
    epoch_rewards = []
    
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes), desc='Training'):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        
        done = False
        while not done:
            action = epsilon_greedy_action_selection(policy_net, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            
            reward -= 0.01
            total_reward += reward
            
            replay_buffer.push(state, action, next_state, reward, done)
            state = next_state
            
            if len(replay_buffer) >= batch_size:
                states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to("cuda")
                actions = torch.LongTensor(actions).to("cuda")
                next_states = torch.FloatTensor(next_states).to("cuda")
                rewards = torch.FloatTensor(rewards).to("cuda")
                dones = torch.FloatTensor(dones).to("cuda")
                
                q_values = policy_net(states)
                next_q_values = target_net(next_states)
                
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = next_q_values.max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.functional.mse_loss(q_values, expected_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_loss += loss.item()
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        epoch_losses.append(episode_loss)
        epoch_rewards.append(total_reward)
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f'policy_net_{episode}.pth')
        
    return policy_net, epoch_losses, epoch_rewards

# %%
def plot_training_statistics(epoch_losses, epoch_rewards):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Reward', color=color)
    ax2.plot(epoch_rewards, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Training Statistics')
    plt.savefig('training_statistics.png')

# %%
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

if __name__ == '__main__':
    ale = ALEInterface()
    ale.loadROM(DoubleDunk)

    # %%
    env = gym.make('ALE/DoubleDunk-ram-v5', obs_type="ram")

    # %%
    num_episodes = 500
    batch_size = 1024
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 25

    policy_net, epoch_losses, epoch_rewards = train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq)

    plot_training_statistics(epoch_losses, epoch_rewards)
        
    torch.save(policy_net.state_dict(), 'policy_net_direct.pth')
    print("Model saved to policy_net.pth")

# %%
# plot_training_statistics(epoch_losses, epoch_rewards)
    
# torch.save(policy_net.state_dict(), 'policy_net_direct.pth')
# print("Model saved to policy_net.pth")


