# We should be able to use the precomputed data as we did in training_torch.py to train this one.

# %%
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from torch.nn.functional import relu


# %%
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

ale = ALEInterface()
ale.loadROM(DoubleDunk)


# no rendering : cpu does the work, faster but massive load on cpu (not recommended)
# render_mode = human : gpu does the work, slower but less load on cpu (safer)
# env = gym.make('ALE/DoubleDunk-v5', obs_type="ram", render_mode="human")

# %%
class DQN(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# %%
def train_dqn(num_episodes, batch_size, gamma, epsilon_start, epsilon_final, epsilon_decay, replay_buffer_capacity):
    env = gym.make('ALE/DoubleDunk-v5', obs_type="ram", render_mode="human")
    
    input_dim = 128
    action_space = 18
    
    policy_net = DQN(input_dim, action_space)
    target_net = DQN(input_dim, action_space)
    
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = 0
        
        while not done or truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state)
                    action = q_values.max(1)[1].item()
                    
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            
            # update q value
            # Compute loss and update Q-network
            # Sample from replay buffer
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Compute Q-values for current and next states
            q_values = policy_net(states)
            q_values_next = target_net(next_states).detach()
            max_q_values_next = q_values_next.max(1)[0]

            # Compute expected Q-values
            expected_q_values = rewards + gamma * (1 - dones.float()) * max_q_values_next

            # Gather Q-values for actions taken
            q_values_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute Huber loss (smooth L1 loss)
            loss = nn.functional.smooth_l1_loss(q_values_action, expected_q_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())


# %%
train_dqn(
    num_episodes=1000,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_final=0.1,
    epsilon_decay=0.999,
    replay_buffer_capacity=10000
)



