import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# Actor network for the Reinforcement Learning agent
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9216 + 1, 512)  # Adjusted to include the timestamp input
        self.fc2 = nn.Linear(512, 2)  # Output: 2 actions (jump, not jump)

    def forward(self, x, timestamp):
        '''
        Forward pass for the actor network. A forward pass is the process of transforming the input data into an output
        using the neural network. In this case, the input is the state and the timestamp, and the output is the action.
        
        Parameters:
            x (tensor): The input tensor representing the state.
            timestamp (tensor): The input tensor representing the timestamp.
            
        Returns:
            tensor: The output tensor representing the action.
        '''
        x = x.view(-1, 9216)
        timestamp = timestamp.view(-1, 1)  # Reshape timestamp to match the batch size
        x = torch.cat((x, timestamp), dim=1)  # Concatenate the timestamp with the state
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Critic network for the Reinforcement Learning agent
class Critic(nn.Module):
   def __init__(self):
       super(Critic, self).__init__()
       self.fc1 = nn.Linear(9216, 512)
       self.fc2 = nn.Linear(512, 1)  # Output: 1 value (state value)

   def forward(self, x):
       '''
       Forward pass for the critic network. A forward pass is the process of transforming the input data into an output 
       using the neural network. In this case, the input is the state, and the output is the state value.
       '''
       x = x.view(-1, 9216)  # Flatten the input tensor
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return x

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, prob_alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = prob_alpha  # Priority exponent for sampling
        self.capacity = capacity  # Maximum buffer size
        self.buffer = deque(maxlen=capacity)  # Deque for storing experiences
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Array for storing priorities
        self.pos = 0  # Current position in the buffer
        self.beta_start = beta_start  # Initial value of the importance sampling weight
        self.beta_frames = beta_frames  # Number of frames for updating the importance sampling weight
        self.frame = 1  # Current frame

    def push(self, state, action, reward, next_state, done, timestamp):
        """
        Add a new experience to the buffer and update the priority.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0  # Maximum priority
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, timestamp))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, timestamp)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with prioritized sampling.
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        prios = prios + 1e-6  # Add a small constant to avoid zero or negative values
        probs = prios ** self.prob_alpha  # Compute probabilities for sampling
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        beta = self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()

        states, actions, rewards, next_states, dones, timestamps = zip(*samples)
        return (np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones), np.stack(timestamps)), indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of the experiences in the buffer.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """
        Return the current length of the buffer.
        """
        return len(self.buffer)