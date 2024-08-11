import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    

def epsilon_greedy_action_selection(policy_net, state, epsilon, device):
    if np.random.rand() < epsilon:
        # return np.random.randint(policy_net.fc5.out_features)
        return np.random.randint(policy_net.net[-1].out_features)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.max(1)[1].item()
    
def plot_training_statistics(epoch_losses, epoch_rewards, filename='training_statistics'):
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
    plt.title('Training Statistics for ' + filename)
    plt.savefig(filename + '.png')