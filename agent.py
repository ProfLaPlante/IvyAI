import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# Convolutional Neural Network (CNN) for the Geometry Dash game
class GeometryDashCNN(nn.Module):
   def __init__(self):
       super(GeometryDashCNN, self).__init__()
       # Convolutional layers
       self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # Input: 1 channel, Output: 32 channels, 8x8 kernel, stride 4
       self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # Input: 32 channels, Output: 64 channels, 4x4 kernel, stride 2
       self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # Input: 64 channels, Output: 64 channels, 3x3 kernel, stride 1

       # Fully connected layers
       self.fc1 = nn.Linear(3840, 512)  # Adjust 3840 to match the actual flattened size
       self.fc2 = nn.Linear(512, 2)  # Output: 2 actions (jump, not jump)

   def forward(self, x):
       # Convolutional layers with ReLU activation
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = F.relu(self.conv3(x))

       # Flatten the tensor for the FC layers
       x = x.view(x.size(0), -1)

       # Fully connected layers with ReLU activation
       x = F.relu(self.fc1(x))
       x = self.fc2(x)

       return x

# Actor network for the Reinforcement Learning agent
class Actor(nn.Module):
   def __init__(self):
       super(Actor, self).__init__()
       self.fc1 = nn.Linear(9216, 512)  # Adjusted to the correct size
       self.fc2 = nn.Linear(512, 2)  # Output: 2 actions (jump, not jump)

   def forward(self, x):
       '''
       Forward pass for the actor network. A forward pass is the process of transforming the input data into an output
       using the neural network. In this case, the input is the state, and the output is the action.
       '''
       x = x.view(-1, 9216)  # Adjusting the view to match the new input size
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

   def push(self, state, action, reward, next_state, done):
       """
       Add a new experience to the buffer and update the priority.
       """
       max_prio = self.priorities.max() if self.buffer else 1.0  # Maximum priority
       if len(self.buffer) < self.capacity:
           self.buffer.append((state, action, reward, next_state, done))
       else:
           self.buffer[self.pos] = (state, action, reward, next_state, done)
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

       probs = prios ** self.prob_alpha  # Compute probabilities for sampling
       probs /= probs.sum()

       indices = np.random.choice(len(self.buffer), batch_size, p=probs)
       samples = [self.buffer[idx] for idx in indices]

       # Compute importance sampling weights
       beta = self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames
       self.frame += 1

       weights = (len(self.buffer) * probs[indices]) ** -beta
       weights /= weights.max()

       return map(np.stack, zip(*samples)), indices, np.array(weights, dtype=np.float32)

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