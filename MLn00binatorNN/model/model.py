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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RegularizedHuberLoss(nn.Module):
    def __init__(self, weight_decay=0.0001234):
        super(RegularizedHuberLoss, self).__init__()
        self.weight_decay = weight_decay
        self.hub_loss = nn.HuberLoss()
        
    def forward(self, logits, targets, model):
        huber_loss = self.hub_loss(logits, targets)
        
        l1_regularization = 0.0
        l2_regularization = 0.0
        for param in model.parameters():
            l1_regularization += torch.norm(param, p=1)
            l2_regularization += torch.norm(param, p=2)

        loss = huber_loss + (self.weight_decay * l1_regularization) + (self.weight_decay * l2_regularization)
        return loss
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ResNet(torch.nn.Module):
    def __init__(self, module):
        super(ResNet, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
        
#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def conv_block(in_f, out_f, repeat_blocks=1, *args, **kwargs):

    block = [nn.Conv2d(in_f, out_f, device=get_device(),*args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm2d(out_f, device=get_device()) ]
        
    sequential_blocks = [ block for i in range(repeat_blocks) ]
    
    return nn.Sequential( *unpack_nested_list(sequential_blocks) )
    
#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ResnetEncoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        
        self.feature_steps = [nn.Conv2d(in_f, out_f, kernel_size=3, padding=1, device=get_device()) for in_f, out_f in zip(enc_sizes, enc_sizes[1:])]
        
        self.resnet_blocks = [ResNet(conv_block(in_f, out_f, repeat_blocks=2, kernel_size=3, padding=1)) for in_f, out_f in zip(enc_sizes[1:], enc_sizes[1:])]
        
        self.max_pool_blocks = [ nn.MaxPool2d(kernel_size=2,stride=2) for i in enc_sizes[:-2] ]
        self.max_pool_blocks.append(torch.nn.AdaptiveAvgPool2d(2))
        
        self.combined_blocks = zip(self.feature_steps, self.resnet_blocks, self.max_pool_blocks)
        self.conv_blocks = nn.Sequential( *unpack_nested_list(self.combined_blocks) )
        
    def forward(self, x):
        return self.conv_blocks(x)
        
#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def decode_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f, device=get_device()),
        nn.ReLU() )
    
#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DeepDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[decode_block(in_f, out_f) for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes, device=get_device())
        
    def forward(self, x):
        return nn.Sequential(self.dec_blocks, self.last)(x)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class QNetworkDCN(nn.Module):
    def __init__(self, init_c=3, enc_sizes=[32,64,128], dec_sizes=[(128*4),128], n_classes=8):
        super().__init__(  )
        
        self.enc_sizes = [init_c, *enc_sizes]
        self.dec_sizes = [ *dec_sizes]
        
        self.online =  nn.Sequential( *[ ResnetEncoder(self.enc_sizes), nn.Flatten(), DeepDecoder(self.dec_sizes, n_classes)] )
    
        # Initialization
        for params in self.online.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
    
        self.target = copy.deepcopy(self.online)
 
         # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
    # -----------------------------------
    def forward(self, grayscale_3panel_img, q_model="online"):
        if q_model == "online":
            return self.online(grayscale_3panel_img)
            
        elif q_model == "target":
            return self.target(grayscale_3panel_img)
        
        
\
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  The default model:
"""
QNetworkDCN(
  (online): Sequential(
    (0): ResnetEncoder(
      (conv_blocks): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ResNet(
          (module): Sequential(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(negative_slope=0.01)
            (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(negative_slope=0.01)
            (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ResNet(
          (module): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(negative_slope=0.01)
            (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(negative_slope=0.01)
            (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ResNet(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(negative_slope=0.01)
            (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(negative_slope=0.01)
            (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (8): AdaptiveAvgPool2d(output_size=2)
      )
    )
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): DeepDecoder(
      (dec_blocks): Sequential(
        (0): Sequential(
          (0): Linear(in_features=512, out_features=128, bias=True)
          (1): ReLU(negative_slope=0.01)
        )
      )
      (last): Linear(in_features=128, out_features=8, bias=True)
    )
  )
"""


