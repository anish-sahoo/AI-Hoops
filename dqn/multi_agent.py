# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

#tdqm imports
from tqdm import tqdm 

# imports for environment
from pettingzoo.atari import double_dunk_v3
from dqn.dqn_model import DeepQNetwork, plot_training_statistics, epsilon_greedy_action_selection
from dqn.replay_buffer import ReplayBuffer

#standard imports
from dataclasses import dataclass

#uses cuda if available
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)

#an example of how to use this code
def sample_usage(device):
    env = double_dunk_v3.env(obs_type="ram", auto_rom_install_path="roms")
    env.reset()
    
    replay_buffer_size = 100
    num_episodes = 125
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99
    target_update_freq = 50

    policy_net, epoch_losses, epoch_rewards = train_dqn(
        env, 
        num_episodes, 
        batch_size, 
        gamma, 
        epsilon_start, 
        epsilon_end, 
        epsilon_decay, 
        target_update_freq, 
        device=get_device(),
        replay_buffer_size=replay_buffer_size

    )

#calls the code
def train_multi_agent():
    device = get_device()
    sample_usage(device)

#A copy of the info used for each agent
@dataclass
class Agent_Info():
    policy_net: any
    target_net: any
    optimizer: any
    replay_buffer: any
    cycle_count: any
    epoch_losses: any
    epoch_rewards: any
    epsilon: any

    
def train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq, 
              negative_reward=-0.01, plot=True, save=True, device="cpu", replay_buffer_size=100):
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
        negative_reward: negative reward for each step
        plot: whether to plot training statistics
        save: whether to save the model
        device: device to run the model on (detects cuda if available by default)
    Returns:
        (policy_net, epoch_losses, epoch_rewards): trained policy network, list of losses for each episode,\n list of total rewards for each episode
    """
    agent_data = {}
    for agent in env.agents:
        input_dim = env.observation_space(agent).shape[0]
        action_space = env.action_space(agent).n
        
        policy_net = DeepQNetwork(input_dim, action_space).to(device)
        target_net = DeepQNetwork(input_dim, action_space).to(device)
        
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters())
        
        replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        cycle_count = 0
        
        epoch_losses = []
        epoch_rewards = []
        
        epsilon = epsilon_start

        agent_data[agent] = Agent_Info(policy_net, target_net, optimizer, replay_buffer, cycle_count, epoch_losses, epoch_rewards, epsilon)

    # loop for every episode
    for episode in tqdm(range(num_episodes), desc='Training'):
        env.reset()
        state = env.last()[0]
        total_reward = 0
        episode_loss = 0
        
        done = False
        while not done:
            for agent in env.agents:
                data = agent_data[agent]
                action = epsilon_greedy_action_selection(data.policy_net, state, data.epsilon, device)
                env.step(action)
                next_state, reward, done, _, _ = env.last()
                
                reward += negative_reward
                total_reward += reward
                
                data.replay_buffer.push(state, action, next_state, reward, done)
                data.cycle_count += 1

                state = next_state
                
                if data.cycle_count >= batch_size:
                    states, actions, next_states, rewards, dones = data.replay_buffer.sample(batch_size)
                    
                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    dones = torch.FloatTensor(dones).to(device)
                    
                    q_values = data.policy_net(states)
                    next_q_values = data.target_net(next_states)
                    
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = next_q_values.max(1)[0]
                    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                    
                    loss = nn.functional.mse_loss(q_values, expected_q_values)
                    
                    data.optimizer.zero_grad()
                    loss.backward()
                    data.optimizer.step()
                    
                    episode_loss += loss.item()
        for agent in env.agents:    
            data = agent_data[agent]
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            data.epoch_losses.append(episode_loss)
            data.epoch_rewards.append(total_reward)
            
            if episode % target_update_freq == 0:
                data.target_net.load_state_dict(data.policy_net.state_dict())
            if episode % 100 == 0 and episode > 0 and save:
                torch.save(data.policy_net.state_dict(), f'policy_net_{episode}_{batch_size}.pth')
    for agent in env.agents:  
        data = agent_data[agent]
        if plot:
            plot_training_statistics(data.epoch_losses, data.epoch_rewards, f'policy_net_{num_episodes}_{batch_size}_{agent}')
        if save:
            torch.save(data.policy_net.state_dict(), f'policy_net_{num_episodes}_{batch_size}_{agent}.pth')
            print(f'Model saved to policy_net_{num_episodes}_{batch_size}_{agent}.pth')
    return (1,1,1)#policy_net, epoch_losses, epoch_rewards


if __name__ == "__main__":
   train_multi_agent()
