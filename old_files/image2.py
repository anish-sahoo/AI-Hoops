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
input()
obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    print(moves[action])
    obs, reward, truncated, info, done = env.step(action)
    
    # Convert observation to grayscale
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to identify players
    thresh = cv2.adaptiveThreshold(gray_obs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use morphological operations to separate overlapping players
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours of the players
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area (this is a simplistic approach)
    player_contours = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 1000]
    
    # Draw contours on the original observation
    cv2.drawContours(obs, player_contours, -1, (0, 255, 0), 3)
    
    # Display the processed frame
    cv2.imshow('Processed Frame', obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    for cnt in player_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f'Player at ({x}, {y}) with width {w} and height {h}')
    
    time.sleep(0.01)

env.close()
cv2.destroyAllWindows()