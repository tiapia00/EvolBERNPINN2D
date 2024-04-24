import torch
import torch.nn as nn
import numpy as np


class NN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, dim_input : int = 3, dim_output : int = 2, act=nn.Tanh()):

        super().__init__()
        self.dim_hidden = dim_hidden
        self.layer_in = nn.Linear(dim_input, self.dim_hidden)
        nn.init.xavier_uniform_(self.layer_in.weight)

        self.num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList()

        for _ in range(self.num_middle):
            middle_layer = nn.Linear(dim_hidden, dim_hidden)
            nn.init.xavier_uniform_(middle_layer.weight)
            self.act = act
            self.middle_layers.append(middle_layer)

        self.layer_out = nn.Linear(dim_hidden, dim_output)
        nn.init.xavier_uniform_(self.layer_out.weight)

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device

class Loss:
    def __init__(
        self,
        X: torch.tensor,
        T: torch.tensor,
        y_true : torch.tensor,
        verbose: bool = False,
    ):
        self.X = X
        self.T = T
        self.y_true = y_true 
        
    def loss(self, nn: NN):
        output = f(nn, self.X, self.T)
        loss = output - self.y_true
        return loss.pow(2).mean()

    def __call__(self, nn: NN):
        """
        Allows you to use instance of this class as if it was a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(nn)
        ```
        """
        return self.loss(nn)

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import pytz
from typing import Callable

def train_model(
    nn_approximator: NN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int
) -> NN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    loss: torch.Tensor = torch.inf

    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):

        loss: torch.Tensor = loss_fn(nn_approximator)
        loss = loss_fn(nn_approximator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        # Log loss
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        pbar.update(1)

    pbar.update(1)
    pbar.close()
    return nn_approximator, np.array(loss_values)

def f(nn: NN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model
    Internally calling the forward method when calling the class as a function"""
    return nn(x, t)

