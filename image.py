import gym
import cv2
import numpy as np
import time

env = gym.make('ALE/DoubleDunk-v5', render_mode='human')

print(env.observation_space)
print(str(env.action_space))
print(env.reward_range)
print(env.metadata)
print(env.spec)
print(env.unwrapped.get_action_meanings())

moves = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

obs = env.reset()
for i in range(500):
    action = env.action_space.sample()
    print(moves[action])
    obs, reward, truncated, info, done = env.step(action)
    
    # Convert observation to grayscale
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to identify players (this is a simplistic approach)
    _, thresh = cv2.threshold(gray_obs, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of the players
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original observation
    cv2.drawContours(obs, contours, -1, (0, 255, 0), 3)
    
    # Display the processed frame
    cv2.imshow('Processed Frame', obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.1)

env.close()
cv2.destroyAllWindows()