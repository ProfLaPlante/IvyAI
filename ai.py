import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from agent import Actor, Critic, PrioritizedReplayBuffer
from screen_capture import capture_screen, preprocess_frame, find_avatar
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt

plt.figure()

avg_time = 0.0
best_time = 0.0

def get_initial_state():
   """
   Capture and preprocess the initial game screen to get the initial state.
   """
   initial_frame = capture_screen()  # Capture the initial game screen
   initial_state = preprocess_frame(initial_frame)  # Preprocess the frame
   return initial_state

# def perform_action(action):
#     if action == 1:  # Assuming 1 corresponds to 'jump'
#         pyautogui.press('space')  # Simulate a spacebar press
#     # No need to explicitly handle the 'not jump' action

def choose_action(state, actor_model, exploration_rate=0.1):
   """
   Choose an action (jump or not jump) based on the current state and the actor model.

   Args:
       state (numpy.ndarray): The current state of the game.
       actor_model (nn.Module): The actor model used for selecting actions.
       exploration_rate (float): The exploration rate for choosing random actions.

   Returns:
       int: The chosen action (0 for 'not jump', 1 for 'jump').
   """
   if np.random.rand() < exploration_rate:
       # Randomly choose an action to explore the environment
       action = np.random.choice([0, 1])  # Assuming 0 is 'not jump' and 1 is 'jump'
   else:
       # Convert state to tensor and add batch dimension (B x C x H x W)
       state_tensor = torch.FloatTensor(state)
       # Forward pass through the actor model to get logits
       logits = actor_model(state_tensor)
       # Get probabilities from logits
       probabilities = torch.softmax(logits, dim=-1)
       # Choose action with the highest probability
       action = torch.argmax(probabilities).item()
   return action

# Dummy function for illustration; you'll need to implement a way to determine rewards
# def calculate_reward():
#     # Implement logic to calculate reward
#     # For now, let's return a placeholder value
#     return 1

def play_step(action, reward, elapsed_time):
   """
   Perform the given action, capture the next state, and calculate the reward and done flag.

   Args:
       action (int): The action to perform (0 for 'not jump', 1 for 'jump').
       reward (float): The current reward.
       elapsed_time (float): The elapsed time since the start of the episode.

   Returns:
       tuple: A tuple containing the next state, next reward, and a flag indicating if the episode is done (player died).
   """
   global best_time
   # Perform the action
   if action == 1:  # Assuming 1 is 'jump'
       # pyautogui.press('space')  # Simulate pressing the spacebar
       pyautogui.click(button='left')  # Simulate a left click
       
   # Wait a bit for the action to take effect in the game
   time.sleep(0.05)
   
   next_reward = reward
   
   # Capture the next state
   next_state = capture_screen()
   next_state_no_expand = preprocess_frame(next_state, False)
   next_state_expanded = preprocess_frame(next_state, True)
   
   # Check if the avatar is found in the next state
   avatar_found = find_avatar(next_state_no_expand)
   
   dead = not avatar_found  # The player has died if the avatar is not found
   
   # Define the reward or penalty
   if dead:
       # Dynamically adjust the reward based on the current reward and the time it took to die
       penalty = -0.5 + (elapsed_time / 200) + (best_time / 1000)  # Dynamically penalizing for dying
       next_reward += penalty
       print(f"Died at {elapsed_time} Penalized: {penalty} points!")
   else:
       # Rationale: the further it progresses in the game, the larger the reward.
       award = 0.2 + (elapsed_time / 200) + (best_time / 1000)  # Dynamically reward the agent for surviving
       next_reward += award
       print(f"Survived for: {elapsed_time} Rewarded: {award}")
       # Small reward for surviving this step
   
   return next_state_expanded, next_reward, dead

def compute_returns(rewards, gamma=0.99):
   """
   Compute the returns for each time step, given the rewards and a discount factor.

   Args:
       rewards (list): A list of rewards.
       gamma (float): The discount factor.

   Returns:
       list: A list of returns, where each element corresponds to the return for that time step.
   """
   returns = []
   R = 0
   for r in reversed(rewards):
       R = r + gamma * R
       returns.insert(0, R)
   return returns

def update_policy(actor_model, critic_model, actor_optimizer, critic_optimizer, states, actions, returns, weights):
   """
   Update the actor and critic models based on the given states, actions, returns, and weights.

   Args:
       actor_model (nn.Module): The actor model.
       critic_model (nn.Module): The critic model.
       actor_optimizer (optim.Optimizer): The optimizer for the actor model.
       critic_optimizer (optim.Optimizer): The optimizer for the critic model.
       states (list): A list of states.
       actions (list): A list of actions.
       returns (list): A list of returns.
       weights (list): A list of importance sampling weights.

   Returns:
       tuple: A tuple containing the actor loss, critic loss, and TD errors.
   """
   actor_optimizer.zero_grad()
   critic_optimizer.zero_grad()
   actor_loss = None
   critic_loss = None
   td_errors = []
   for state, action, R, weight in zip(states, actions, returns, weights):
       state = torch.tensor(state, dtype=torch.float)
       action = torch.tensor(action, dtype=torch.long)
       R = torch.tensor(R, dtype=torch.float)
       weight = torch.tensor(weight, dtype=torch.float)

       # Check if the state is 3D or 4D
       if len(state.shape) == 3:
           # Forward pass to get log probabilities from actor
           logits = actor_model(state.unsqueeze(0))  # Add batch dimension
           log_probs = F.log_softmax(logits, dim=-1)
       else:  # 4D
           logits = actor_model(state)
           log_probs = F.log_softmax(logits, dim=-1)

       # Select the log probability for the taken action
       log_prob_action = log_probs[0, action]

       # Calculate the actor loss
       actor_loss -= weight * log_prob_action * R  # Negative because we want to do gradient ascent

       # Forward pass to get value estimate from critic
       value_estimate = critic_model(state.unsqueeze(0) if len(state.shape) == 3 else state)

       # Calculate the critic loss (MSE between the return and the value estimate)
       critic_loss += weight * F.mse_loss(value_estimate, R.unsqueeze(0).unsqueeze(1))

       # Compute TD error
       td_error = R - value_estimate.detach()
       td_errors.append(td_error)

   # Backward pass and optimization step for actor
   if actor_loss is not None:
       actor_loss.backward()
       actor_optimizer.step()

   # Backward pass and optimization step for critic
   if critic_loss is not None:
       critic_loss.backward()
       critic_optimizer.step()
       
   if len(td_errors) > 0:
       td_errors_stacked = torch.stack(td_errors)
   else:
        # Handle the empty case by creating a default tensor or skipping operations that require td_errors
        td_errors_stacked = torch.tensor([])

   return actor_loss.item() if actor_loss is not None else 0, critic_loss.item() if critic_loss is not None else 0, td_errors_stacked

def is_avg(current_time):
    """
    Check if the current time is greater than or equal to the average time.

    Args:
        current_time (float): The current time.

    Returns:
        bool: True if the current time is greater than or equal to the average time, False otherwise.
    """
    global avg_time
    if current_time >= avg_time:
        return True
    return False

def is_new_best(current_time):
    """
    Check if the current time is a new best time, and update the best time if so.

    Args:
        current_time (float): The current time.

    Returns:
        bool: True if the current time is a new best time, False otherwise.
    """
    global best_time
    if current_time >= best_time:
        best_time = current_time
        return True
    return False

def train_simple_rl(actor_model, critic_model, episodes, gamma=0.99):
    """
    Train the actor and critic models using a simple Reinforcement Learning algorithm.

    Args:
        actor_model (nn.Module): The actor model.
        critic_model (nn.Module): The critic model.
        episodes (int): The number of episodes to train for.
        gamma (float): The discount factor for computing returns.

    Returns:
        tuple: A tuple containing lists of actor losses and critic losses.
    """
    global avg_time
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.01)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.01)
    reward = 0.0
    epsilon = 10
    actor_losses = []
    critic_losses = []
    prev_times = []

    start_global_time = time.time()
    
    # Set up the reply buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for episode in range(episodes):
        states, actions, rewards = [], [], []
        state = get_initial_state()
        dead = False
        
        start_time = time.time()
        
        # Decrease epsilon over time
        next_epsilon = max(0.05, epsilon * (1 / (episode + 1)))
        
        # Primer so that no error is thrown when the first action is taken
        action = choose_action(state, actor_model, next_epsilon)
        next_state, next_reward, dead = play_step(action, reward, 0.0)
        reward = next_reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        current_time = 0.0
        
        while not dead:
            current_time = time.time() - start_time
            print("current_time", current_time)
            
            action = choose_action(state, actor_model)
            next_state, next_reward, dead = play_step(action, reward, current_time)
            # Subtract a tiny amount from the reward for taking too long to beat the level
            next_reward -= (time.time() - start_global_time) * 0.001
            reward = next_reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        if is_new_best:  # We know there is a new best time
            replay_buffer.push(state, action, reward, next_state, dead)
            print("New best time:", best_time)
            next_reward += 5.0
        else:
            next_reward -= 2.0

        if is_avg:
            next_reward += 2.0
        else:
            next_reward -= 5.0
            
        print("Reward:", reward, "Dead:", dead, "Episode:", episode)

        start_time = time.time()
        
        if len(prev_times) != 4:
            prev_times.append(current_time)
        else:
            prev_times.pop(0)  # Remove the first element
            avg_time = sum(prev_times) / len(prev_times)
            print("Adding 1st time to prev_times", current_time)
            prev_times.append(current_time)
            
        # If the replay buffer is full, sample from it and train the model
        batch_size = 32  # Define the batch size
        if len(replay_buffer) > batch_size:
            experiences, indices, weights = replay_buffer.sample(batch_size)
            returns = compute_returns([exp[2] for exp in experiences], gamma)
            actor_loss, critic_loss, td_errors = update_policy(actor_model, critic_model, optimizer_actor, optimizer_critic, [exp[0] for exp in experiences], [exp[1] for exp in experiences], returns, weights)

            # Update priorities with TD errors
            replay_buffer.update_priorities(indices, td_errors.numpy())

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if episode % 5 == 0:
            for i in range(len(prev_times) - 1):
                difference = abs(prev_times[i] - prev_times[i + 1])
                print(f"Difference: {difference}")
                if 0.5 <= difference <= 2:
                    next_reward -= 5.0
                    print(f"Distances are between 0.02 and 0.04! New reward amt: {next_reward}")
                    break
        
        if episode % 10 == 0:
            # Print a highly formatted list of the current reward and penalty
            print(f"Episode: {episode}, Total reward: {sum(rewards)}, Total penalty: {len(rewards) - sum(rewards)}")
            next_reward = (next_reward / (episode + 1))

    return actor_losses, critic_losses


def run_inference(actor_model):
    """
    Run the actor model in inference mode for playing the game.

    Args:
        actor_model (nn.Module): The actor model to use for inference.
    """
    state = get_initial_state()
    dead = False
    reward = 0.0

    try:
        while True:
            start_time = time.time()
            action = choose_action(state, actor_model)
            next_state, next_reward, dead = play_step(action, reward, time.time() - start_time)
            reward = next_reward
            state = next_state
            print("Reward:", reward, "Dead:", dead)

            if dead:
                state = get_initial_state()
                dead = False
                reward = 0.0
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("Inference stopped by user.")

def main():
    # Ask the user if they want to use a pre-trained model or train a new one
    pre_trained_input = input("Do you want to use a pre-trained model? (yes/no): ")
    
    if pre_trained_input.lower() == "yes":
        # Load the pre-trained model
        # Check if a pre-trained model exists
        try:
            actor_model = Actor()
            critic_model = Critic()
            # Load the trained models
            checkpoint = torch.load(r"checkpoints/geometry_dash_model.pt")
            actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
            critic_model.load_state_dict(checkpoint['critic_model_state_dict'])

            # Ask the user if they want to use the model for inference or continue training
            mode_input = input("Do you want to use the model for inference or continue training? (inference/training): ")
            if mode_input.lower() == "training":
                actor_model.train()  # Switch the actor model to training mode
                critic_model.train()  # Switch the critic model to training mode
                actor_losses, critic_losses = train_simple_rl(actor_model, critic_model, episodes=1000)  # Continue training the model
            else:
                actor_model.eval()  # Switch the actor model to evaluation mode
                critic_model.eval()  # Switch the critic model to evaluation mode
                run_inference(actor_model)  # Use the actor model for inference
        except FileNotFoundError:
            print("Pre-trained model not found. Training a new model...")
            actor_model = Actor()
            critic_model = Critic()
            actor_losses, critic_losses = train_simple_rl(actor_model, critic_model, episodes=1000)
    else:
        # Train a new model
        actor_model = Actor()
        critic_model = Critic()
        actor_losses, critic_losses = train_simple_rl(actor_model, critic_model, episodes=1000)
    
    # Plot the losses
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
        
    # Save the trained models
    torch.save({
        'actor_model_state_dict': actor_model.state_dict(),
        'critic_model_state_dict': critic_model.state_dict(),
    }, r"checkpoints/geometry_dash_model.pt")

if __name__ == "__main__":
    main()