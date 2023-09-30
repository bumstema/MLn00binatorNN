MLn00binatorNN
==============
A neural network trained to play emulated games (Pokemon Yellow) using reinforcement Q learning.
The model uses centered-action-colour images, where each channel (r,g,b) is ihnstead represented by the grayscale screenshot frame of the before-action-after game states.


![RunTrainingTrajectories_(level-1_ep_5)](https://github.com/bumstema/MLn00binatorNN/assets/25807978/e24449de-9bb8-4e1d-8879-89a172d8d2a8)
Future states appear blue-shifted and previous states appear red-shifted when interpreted as colour channels.
The green channel is the frame as the action is happening.
This animation shows 5 learning episodes, with the truncation goal being set as the staircase in the right top corner.



The neural network optimizes the playthrough to minimize the number of frames taken to reach a goal.
It is designed to facilitate learning a style of gameplay that is reminescent of speed runners.

Rom files are loaded with PyBoy. (not included)

__`pip install pyboy`__

The environment contains the game's state and the agent.
The agent is the neural network who acts in the environment based on it's expected reward which produces a new state.
Most of the flow was inspired from the Pytorch Tutorials on [Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) and [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

Start off by loading the environment.

```python
from MLn00binatorNN.environment.env import GameWrapperEnvironmentPokemon

environment = GameWrapperEnvironmentPokemon()
environment.start_game(f'{GAME_PATH}', n_training_steps, train=True, save_frames=False)
```    

Truncation states are defined using hashlib.sha256 to hash the state's flattened pixel array.
This allows for the compact enumeration of state's that have been visited before.

Add these to: termination.py 

Tune training loop variables in env.py 

Set learning update parameters in q_agent.py 

Directory variables can be set in const.py


