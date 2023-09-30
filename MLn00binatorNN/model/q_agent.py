# Include Debug and System Tools
import traceback
import sys, os, os.path
import transformers
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import namedtuple, deque
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Function
import torchvision.models as models
import torchvision


from ..data_io.utils import process_raw_frame, unpack_nested_list, image_to_tensor, get_device
from ..environment.controls import BUTTONS
from .model import QNetworkDCN, RegularizedHuberLoss


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class QLearningAgent:

    # ---- (INIT) ----     # -----------------------------------
    def __init__(self, learning_rate=0.001337, gamma=0.95):
        self.q_network = QNetworkDCN()
        print(self.q_network)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate, momentum=0.85)
        self.loss_fn = RegularizedHuberLoss()
        self.gamma = gamma

        self.burnin = 1_000  # min. experiences before training
        self.learn_every = 128  # no. of experiences between updates to Q_online
        self.sync_every = 1_000  # no. of experiences between Q_target & Q_online sync

        self.batch_size = 32
        self.max_mem = 1_000_000
        self.memory = deque([], maxlen=self.max_mem)
        self.accepted_memory = deque([], maxlen=self.max_mem)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.nonrandom_decisions = 0

        self.save_every = 1e4  # no. of experiences between saving NNet

        self.episode_steps = 99999
        self.least_episode_steps = 99999
        
    # ---- (ACT) ----     # -----------------------------------
    def select_action(self, state, epsilon = 0.1):
    
        if random.random() < self.exploration_rate:   # Explore
        
            # decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

            return random.randint(0, len(BUTTONS)- 1), random.choice([-1,0])
        
        else:   # Exploit
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=get_device())
                action_values = self.q_network(state_tensor, q_model="online")
                action_index = action_values.argmax(dim=1).item()
                reward = (action_values[0][action_index]).item()
                
                self.nonrandom_decisions += 1
                del state_tensor, action_values
                return action_index, reward

    # ---- (CACHE) ----     # -----------------------------------
    def cache(self, state, action, reward, next_state, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        next_state (``LazyFrame``),
        done(``bool``))
        """

        if len(self.memory) == self.max_mem: self.memory.popleft()
        self.memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})
        
        del state, action, reward, next_state, done
        return
    
    # ----------------------------------- # -----------------------------------
    def update_cache_for_fastest_runs(self):

        if self.episode_steps < self.least_episode_steps:
            if random.random() > 0.5:
                self.least_episode_steps = self.episode_steps
            self.accepted_memory = copy.deepcopy(self.memory)
            if self.episode_steps  < 100: self.learn_every = 4
            if self.episode_steps  < 50: self.learn_every = 2

            else:
                self.learn_every = self.learn_every // 2
            print(f"{self.learn_every = } | {len(self.accepted_memory) = } | {self.exploration_rate = :.4f} | {self.nonrandom_decisions = }")
            self.burnin = len(self.accepted_memory)
            self.sync_Q_target()
        else:
            self.accepted_memory = copy.deepcopy(self.memory)
            
    # ---- (RECALL) ----     # -----------------------------------
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.accepted_memory, self.batch_size)
        
        state,  action, reward, next_state, done = zip(*[[b.get(key) for key in ["state", "action", "reward", "next_state", "done"]] for b in batch])
 
        state = torch.tensor(np.array(state), dtype=torch.float32, device=get_device())
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=get_device())
        action = torch.tensor(list(action), device = get_device())
        reward = torch.tensor(list(reward), device = get_device())
        done = torch.tensor(list(done), device = get_device())
        
        return state, action, reward, next_state, done

    # ---- (LEARN) ----     # -----------------------------------
    def learn(self):
    
        if self.episode_steps == 0 :
            self.sync_Q_online()

        #if self.curr_step < self.burnin:
        if len(self.accepted_memory) < self.burnin:
            return None, None

        if ((len(self.accepted_memory) + self.episode_steps) % self.learn_every) != 0:
            return None, None

        # Sample from memory
        state, action, reward, next_state, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        del state, action, reward, next_state, done
        return td_est.mean().item(), loss

    # -----------------------------------
    def td_estimate(self, state, action):

        current_Q = self.q_network(state, q_model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    # -----------------------------------
    #@torch.no_grad()
    def td_target(self, reward, next_state, done):
        with torch.no_grad():
            next_state_Q = self.q_network(next_state, q_model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.q_network(next_state, q_model="target")[
                np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    # -----------------------------------
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target, self.q_network)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
  
    # -----------------------------------
    def sync_Q_target(self):
        self.q_network.target.load_state_dict(self.q_network.online.state_dict())

    # -----------------------------------
    def sync_Q_online(self):
        self.q_network.online.load_state_dict(self.q_network.target.state_dict())

    # -----------------------------------
    def save_model(self, file_name_path_model_state_dict):
        torch.save(self.q_network.state_dict(), file_name_path_model_state_dict )

    # -----------------------------------
    def load_model(self, file_name_path_model_state_dict ):
        self.q_network.load_state_dict(torch.load(file_name_path_model_state_dict))

    # -----------------------------------# -----------------------------------
    # -----------------------------------# -----------------------------------

