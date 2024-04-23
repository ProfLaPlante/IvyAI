# Geometry Dash Reinforcement Learning Agent

This project implements a Reinforcement Learning (RL) agent designed to play the game Geometry Dash. The agent uses a Convolutional Neural Network (CNN) to process visual inputs and determine the best actions to take within the game environment.

## Project Structure

- `bot.py`: The main script containing the RL loop where the agent learns to interact with the game.
- `screen_capture.py`: Contains image preprocessing functionality and screenshot capturing from the game.
- `agent.py`: Defines the DQN agent and related functionality, such as the model architecture and the replay buffer.
- `requirements.txt`: Lists all Python libraries needed to run the agent.

## Setup

To get started, clone this repository and install the required Python dependencies:

```bash
git clone https://github.com/ProfLaPlante/IvyAI/tree/Geometry-Dash-AI.git
cd geometry-dash-rl-agent
pip install -r requirements.txt
```

## Running the Agent

To start training the agent, simply run:

```bash
python ai.py
```

The script will automatically begin capturing frames from the game, preprocess them, and feed them to the agent to make decisions and learn through reinforcement learning.

## Image Processing

The `screen_capture.py` script preprocesses the game frames for the agent to consume. It handles grayscale conversion, resizing to lower dimensions, and normalization of pixel values.

## Actor Critic Agent

The `agent.py` script defines the Actor Critic agent, including the neural network model, action selection, and learning mechanisms. It uses experience replay and a target network based on the Actor Critic algorithm to stabilize training and improve learning efficiency.

## Contribution

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the GNU General Public License. See `LICENSE` for more information.

## Contact

No contact info yet (Stay Tuned!)
