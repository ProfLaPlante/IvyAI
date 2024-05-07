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

def choose_action(state, actor_model, timestamp, exploration_rate=0.1):
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
		action = np.random.choice([0, 1])
	else:
		state_tensor = torch.FloatTensor(state)
		timestamp_tensor = torch.FloatTensor([timestamp])  # Convert timestamp to a tensor
		logits = actor_model(state_tensor, timestamp_tensor)
		probabilities = torch.softmax(logits, dim=-1)
		action = torch.argmax(probabilities).item()
	return action

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

	return next_state_expanded, next_reward, dead, elapsed_time

def compute_returns(rewards, gamma=0.99):
	"""
	Compute the returns for each time step, given the rewards and a discount factor.
	This function calculates the return for each time step as the sum of future rewards.
	This is necessary because the rewards are delayed and the agent needs to consider future rewards.

	How the math works:
	- Return at time t: R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
	- Return at time t (in general form): R_t = sum_{i=0}^{T-t} (gamma^i * r_{t+i})

	where:
	- r_t is the reward at time t
	- gamma is the discount factor
	- T is the total number of time steps
	- R_t is the return at time t

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

def update_policy(actor_model, critic_model, actor_optimizer, critic_optimizer, states, actions, returns, timestamps, weights):
	"""
	Update the actor and critic models based on the given states, actions, returns, and weights.

	How it works internally:
	- The actor model is updated using policy gradients, where the loss is the negative log probability of the action taken multiplied by the return.
	- The critic model is updated using the mean squared error (MSE) loss between the return and the value estimate.
	- The actor and critic models are updated using the respective optimizers.
	- The TD errors are computed and returned for updating the priorities in the replay buffer.

	How the math works:
	- Actor loss: L_actor = -w * log(p(a|s)) * R
	- Critic loss: L_critic = w * (R - V(s))^2
	- TD error: TD_error = R - V(s)

	where:
	- L_actor is the actor loss
	- L_critic is the critic loss
	- w is the importance sampling weight
	- p(a|s) is the probability of taking action a in state s
	- R is the return
	- V(s) is the value estimate of state s
	- TD_error is the temporal difference error

	Note: The negative sign in the actor loss is because we want to maximize the log probability of the action taken.

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
	actor_loss = torch.tensor(0.0, requires_grad=True)  # Initialize actor_loss as a tensor
	critic_loss = None
	td_errors = []
	for state, action, R, timestamp, weight in zip(states, actions, returns, timestamps, weights):
		state = torch.tensor(state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		R = torch.tensor(R, dtype=torch.float)
		timestamp = torch.tensor(timestamp, dtype=torch.float)
		weight = torch.tensor(weight, dtype=torch.float)

		if len(state.shape) == 3:
			logits = actor_model(state.unsqueeze(0), timestamp.unsqueeze(0))
			log_probs = F.log_softmax(logits, dim=-1)
		else:  # 4D
			logits = actor_model(state, timestamp)
			log_probs = F.log_softmax(logits, dim=-1)

		log_prob_action = log_probs[0, action]
		actor_loss = actor_loss - weight * log_prob_action * R  # Create a new tensor for the actor loss

		value_estimate = critic_model(state.unsqueeze(0) if len(state.shape) == 3 else state)

		critic_loss = weight * F.mse_loss(value_estimate, R.unsqueeze(0).unsqueeze(1)) if critic_loss is None else critic_loss + weight * F.mse_loss(value_estimate, R.unsqueeze(0).unsqueeze(1))

		td_error = R - value_estimate.detach()
		td_errors.append(td_error)

	actor_loss.backward()  # Backward pass for actor
	actor_optimizer.step()

	critic_loss.backward()  # Backward pass for critic
	critic_optimizer.step()

	if len(td_errors) > 0:
		td_errors_stacked = torch.stack(td_errors)
	else:
		td_errors_stacked = torch.tensor([])

	return actor_loss.item(), critic_loss.item(), td_errors_stacked

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
	
	How it works:
	- The agent interacts with the environment by choosing actions and receiving rewards.
	- The actor model selects actions based on the current state.
	- The critic model estimates the value of the state.
	- The agent receives rewards and updates the models based on the rewards.
	- The models are trained using policy gradients and the TD error.
	- The replay buffer is used to store experiences and sample batches for training.
	
	How the math works (to compute the rewards):
	- Reward at time t: r_t = 1 if survived, -1 if died
	- Discount factor: gamma (0.99)
	- Return at time t: R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
	
	where:
	- r_t is the reward at time t
	- gamma is the discount factor
	- R_t is the return at time t
	
	Note: The return is the sum of future rewards, discounted by the discount factor.

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
	reward = 0.0 # Reward for the current step
	epsilon = 10 # Exploration rate
	actor_losses = []
	critic_losses = []
	prev_times = []
 
	penalty_counts = {}  # Dictionary to store penalty counts for each timestamp
	obstacles = set()  # Set to store identified obstacles (timestamps)
	successful_jumps = set()  # Set to store successful jumps (timestamps)
	penalty_threshold = 5  # Threshold for considering a timestamp as an obstacle

	start_global_time = time.time()
	
	# Set up the reply buffer
	replay_buffer = PrioritizedReplayBuffer(capacity=1000)
	
	for episode in range(episodes):
		states, actions, rewards, timestamps, dones = [], [], [], [], []
		state = get_initial_state()
		dead = False
		
		start_time = time.time()
		
		# Decrease epsilon over time
		next_epsilon = max(0.05, epsilon * (1 / (episode + 1)))
		
		# Primer so that no error is thrown when the first action is taken
		action = choose_action(state, actor_model, next_epsilon)
		next_state, next_reward, dead, elapsed_time = play_step(action, reward, 0.0)
		reward = next_reward
		states.append(state)
		actions.append(action)
		rewards.append(reward)
		timestamps.append(elapsed_time)
		dones.append(dead)
		state = next_state
		current_time = 0.0
		
		while not dead:
			current_time = time.time() - start_time
			print("current_time", current_time)
			
			action = choose_action(state, actor_model, elapsed_time)
			next_state, next_reward, dead, elapsed_time = play_step(action, reward, current_time)  # Get the elapsed time from play_step
			# ...
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			timestamps.append(elapsed_time)  # Append the elapsed time to the timestamps list
			
			state = next_state
			
		# Store the experiences in the replay buffer
		for state, action, reward, timestamp, dead in zip(states, actions, rewards, timestamps, dones):
			replay_buffer.push(state, action, reward, next_state, dead, timestamp)  # Store the timestamp in the replay buffer

		if is_new_best:  # We know there is a new best time
			replay_buffer.push(state, action, reward, next_state, dead, elapsed_time)
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
			states, actions, rewards, next_states, dones, timestamps = experiences  # Unpack the experiences
			returns = compute_returns(rewards, gamma)
			actor_loss, critic_loss, td_errors = update_policy(actor_model, critic_model, optimizer_actor, optimizer_critic, states, actions, returns, timestamps, weights)

			# Update priorities with TD errors
			replay_buffer.update_priorities(indices, td_errors.numpy())

			actor_losses.append(actor_loss)
			critic_losses.append(critic_loss)
			
			# Analyze the timestamps and their associated rewards/penalties
			for timestamp, reward, action in zip(list(timestamps), rewards, actions):
				if reward < 0:  # Penalty
					# Increment the penalty count for this timestamp
					penalty_counts[timestamp] = penalty_counts.get(timestamp, 0) + 1
					
					# If the penalty count exceeds the threshold, consider it an obstacle
					if penalty_counts[timestamp] >= penalty_threshold:
						obstacles.add(timestamp)
				else:  # Reward
					# If the timestamp corresponds to an identified obstacle and the action was a jump, consider it a successful jump
					if timestamp in obstacles and action == 1:
						successful_jumps.add(timestamp)

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