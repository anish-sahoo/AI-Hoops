# imports
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 

# local imports
from dqn_model import DeepQNetwork, plot_training_statistics, epsilon_greedy_action_selection
from replay_buffer import ReplayBuffer

# imports for environment
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

def train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq, negative_reward=-0.01, plot=True, save=True, device=device):
    """
    Training function for DQN
    Args:
        env: any gym environment, should work for any observation type that returns a numpy array (ram is preferred)
        num_episodes: number of episodes to train the agent
        batch_size: batch size for training
        gamma: discount factor
        epsilon_start: initial epsilon value for epsilon-greedy action selection
        epsilon_end: final epsilon value for epsilon-greedy action selection
        epsilon_decay: decay rate for epsilon
        target_update_freq: frequency of updating target network
        negative_reward: negative reward for each step, uses -0.01 by default
        plot: whether to plot training statistics
        save: whether to save the model
        device: device to run the model on (detects cuda if available by default)
    Returns:
        (policy_net, epoch_losses, epoch_rewards): trained policy network, list of losses for each episode,\n list of total rewards for each episode
    """
    input_dim = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    policy_net = DeepQNetwork(input_dim, action_space).to(device)
    target_net = DeepQNetwork(input_dim, action_space).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    
    replay_buffer = ReplayBuffer(capacity=500)
    
    epoch_losses = []
    epoch_rewards = []
    
    epsilon = epsilon_start
    cycle_count = 0
    
    for episode in tqdm(range(num_episodes), desc='Training'):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        done = False
        while not done:
            action = epsilon_greedy_action_selection(policy_net, state, epsilon, device)
            next_state, reward, done, _, _ = env.step(action)
            
            # reward += negative_reward
            modified_reward = reward*1000 + negative_reward
            total_reward += modified_reward
            # total_reward += reward
            
            replay_buffer.push(state, action, next_state, modified_reward, done)
            # replay_buffer.push(state, action, next_state, reward, done)
            state = next_state
            cycle_count += 1
            # if len(replay_buffer) >= batch_size:
            if cycle_count >= batch_size:
                states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
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
        if episode % 100 == 0 and episode > 0 and save:
            torch.save(policy_net.state_dict(), f'policy_net_{episode}_{batch_size}_{negative_reward}.pth')
            if plot:
                plot_training_statistics(epoch_losses, epoch_rewards, f'policy_net_{episode}_{batch_size}_{negative_reward}')
    if plot:
        plot_training_statistics(epoch_losses, epoch_rewards, f'policy_net_{num_episodes}_{batch_size}_{negative_reward}')
    if save:
        torch.save(policy_net.state_dict(), f'policy_net_{num_episodes}_{batch_size}_{negative_reward}.pth')
        print(f'Model saved to policy_net_{num_episodes}_{batch_size}_{negative_reward}.pth')
    return policy_net, epoch_losses, epoch_rewards

# Example usage
if __name__ == '__main__':
    ale = ALEInterface()
    ale.loadROM(DoubleDunk)

    env = gym.make('ALE/DoubleDunk-ram-v5', obs_type="ram")

    num_episodes = 2000
    batch_size = 32
    gamma = 0.995
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10

    policy_net, epoch_losses, epoch_rewards = train_dqn(
        env, 
        num_episodes, 
        batch_size, 
        gamma, 
        epsilon_start, 
        epsilon_end, 
        epsilon_decay, 
        target_update_freq, 
        device="cuda",
        negative_reward=-0.1
    )