# imports
import numpy as np
import random

class ReplayBuffer:
    """
    This class instantiates a replay buffer that is used to store the
    state, action, and next state, reward and done transitions. This
    allows us to sample a random batch of transitions within the game
    state of the environment.
    
    """
    def __init__(self, capacity):
        """
        This function initializes the Replay Buffer
        with a specified capacity.

        params:
            capacity(int): number of transitions to 
                           instantiate the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """
        This function adds a transition to the buffer.

        params:
            state: the current state of the game.
            action: the selected action.
            next_state: the next game state observation after the action.
            reward(float): the reward after the action.
            done(boolean): game termination condition.
        """
        state = np.array(state)
        next_state = np.array(next_state)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        This function randomly samples a batch of transitions
        from the Replay Buffer.

        params:
            batch_size(int): number of transitions to sample

        returns:
            a tuple of arrays for the state, action, next_state, 
            reward and done states
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, done

    def __len__(self):
        """
        This function gets the size of the buffer.

        returns:
            the size of the buffer (int).
        """
        return len(self.buffer)
