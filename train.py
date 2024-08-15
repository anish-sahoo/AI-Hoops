import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import DoubleDunk

from dqn.dqn_training import train_dqn

if __name__ == '__main__':
    ale = ALEInterface()
    ale.loadROM(DoubleDunk)

    env = gym.make('ALE/DoubleDunk-ram-v5', obs_type="ram")

    num_episodes = 2000
    batch_size = 32
    gamma = 0.995
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.996
    target_update_freq = 40

    policy_net, epoch_losses, epoch_rewards = train_dqn(
        env=env, 
        num_episodes=num_episodes, 
        batch_size=batch_size, 
        gamma=gamma, 
        epsilon_start=epsilon_start, 
        epsilon_end=epsilon_end, 
        epsilon_decay=epsilon_decay, 
        target_update_freq=target_update_freq, 
        save_interval=500,
        replay_buffer_size=1000,
        device="cuda",
        negative_reward=-0.01,
    )