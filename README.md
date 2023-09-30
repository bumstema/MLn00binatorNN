# MLn00binatorNN
A neural network trained to play emulated games (Pokemon Yellow) using reinforcement Q learning.


Rom files are loaded with PyBoy. (not included)

```python
pip install pyboy
```

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

