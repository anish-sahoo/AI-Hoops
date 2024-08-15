import polars as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import relu
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset

torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayDataset(Dataset):
    def __init__(self, states, actions, next_states, rewards, dones):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.next_states = torch.FloatTensor(next_states)
        self.rewards = torch.FloatTensor(rewards)
        self.dones = torch.FloatTensor(dones)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.dones[idx])

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_space)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_dqn(dataloader, num_epochs, gamma, target_update_freq):
    input_dim = 128
    action_space = 18
    
    policy_net = DeepQNetwork(input_dim, action_space)
    target_net = DeepQNetwork(input_dim, action_space)
    
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    
    epoch_losses = []
    epoch_rewards = []
    
    for epoch in tqdm(range(num_epochs), desc='Training'):
        epoch_loss = 0
        total_reward = 0
        for states, actions, next_states, rewards, dones in dataloader:
            q_values = policy_net(states)
            next_q_values = target_net(next_states)
            
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            loss = nn.functional.mse_loss(q_values, expected_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_reward += rewards.sum().item()
        
        epoch_losses.append(epoch_loss / len(dataloader))
        epoch_rewards.append(total_reward / len(dataloader.dataset))
        
        if epoch % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
    return policy_net, epoch_losses, epoch_rewards

def plot_training_statistics(epoch_losses, epoch_rewards):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  
    # color = 'tab:blue'
    # ax2.set_ylabel('Total Reward', color=color)
    # ax2.plot(epoch_rewards, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Training Statistics')
    plt.savefig('training_statistics.png')

def get_action_probabilities(policy_net, state):
    policy_net.eval()
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        action_probabilities = torch.nn.functional.softmax(q_values, dim=1)
    return action_probabilities.squeeze().numpy()

def load_data(size=100, folders=1):
    replay_buffer_df = pd.DataFrame()
    for j in range(0, folders):
        for i in tqdm(range(0, size), desc=f'Loading data from folder {str(j)}'):
            file_path = f'data/data{j}_compressed/data{str(i)}.json'
            df = pd.read_json(file_path)
            replay_buffer_df = pd.concat([replay_buffer_df, df])
    return replay_buffer_df

if __name__ == '__main__':
    replay_buffer_df = load_data(1000)
    states = np.stack(replay_buffer_df['state'].to_numpy())
    actions = replay_buffer_df['action'].to_numpy()
    next_states = np.stack(replay_buffer_df['new_state'].to_numpy())
    rewards = replay_buffer_df['reward'].to_numpy()
    dones = replay_buffer_df['done'].to_numpy()

    dataset = ReplayDataset(states, actions, next_states, rewards, dones)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)

    policy_net, epoch_losses, epoch_rewards = train_dqn(dataloader, num_epochs=100, gamma=0.99, target_update_freq=10)
    
    plot_training_statistics(epoch_losses, epoch_rewards)
    
    torch.save(policy_net.state_dict(), 'policy_net.pth')
    print("Model saved to policy_net.pth")