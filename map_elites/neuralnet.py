"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies

Structure and main functions for basic single layer linear Neural Network
"""
import torch
from torch import nn, from_numpy
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hid_size, out_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hid_size),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Linear(hid_size, out_size),
            nn.Sigmoid(),
        )
        self.model.requires_grad_(False)
        self.input_size = input_size
        self.hid = hid_size
        self.act_size = out_size
        self.w0_size = self.input_size * self.hid
        self.w2_size = self.hid * self.act_size
        self.b0_size = self.hid
        self.b2_size = self.act_size

    def run(self, x):
        return self.model(x)

    def get_weights(self):
        d = self.model.state_dict()
        return [d['0.weight'], d['2.weight']]

    def get_biases(self):
        d = self.model.state_dict()
        return [d['0.bias'], d['2.bias']]

    def get_model(self):
        return self.model.state_dict()

    def set_trained_network(self, x):
        # Where to slice the array for the weights and biases
        cut0 = self.b0_size
        cut1 = cut0 + self.b2_size
        cut2 = cut1 + self.w0_size

        # Use this block to set the weights AND the biases. Like a real puppet.
        b0_wts = from_numpy(np.array(x[:cut0]))
        b1_wts = from_numpy(np.array(x[cut0:cut1]))
        w0_wts = from_numpy(np.reshape(x[cut1:cut2], (self.hid, self.input_size)))
        w2_wts = from_numpy(np.reshape(x[cut2:], (self.act_size, self.hid)))

        self.set_biases([b0_wts, b1_wts])
        self.set_weights([w0_wts, w2_wts])

    def set_weights(self, weights):
        d = self.model.state_dict()
        d['0.weight'] = weights[0]
        d['2.weight'] = weights[1]
        self.model.load_state_dict(d)

    def set_biases(self, biases):
        d = self.model.state_dict()
        d['0.bias'] = biases[0]
        d['2.bias'] = biases[1]
        self.model.load_state_dict(d)

    def forward(self, x):
        x = torch.Tensor(x)
        flat_x = torch.flatten(x)
        logits = self.model(flat_x)
        return logits



