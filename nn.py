import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import pytz
from typing import Callable

class NN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as a universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, dim_input: int = 3, dim_output: int = 2, act=nn.Tanh()):

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

class Loss_NN:
    def __init__(
        self,
        X: torch.tensor,
        T: torch.tensor,
        y_true: torch.tensor,
        verbose: bool = False,
    ):
        self.X = X
        self.T = T
        self.y_true = y_true

    def loss(self, nn: NN):
        output = f_nn(nn, self.X, self.T)
        loss = output - self.y_true
        return loss.pow(2).mean()

    def __call__(self, nn: NN):
        return self.loss(nn)

def train_model_nn(
    nn_approximator: NN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    x_val: torch.tensor,
    t_val: torch.tensor,
    y_val: torch.tensor,
    path: str,
) -> NN:
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    pbar = tqdm(total=max_epochs, desc="Training", position=0)
    log_dir = f'{path}/logs'
    
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(max_epochs):
        loss = loss_fn(nn_approximator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

        output = f_nn(nn_approximator, x_val, t_val)
        loss_val = output - y_val
        writer.add_scalar("Global loss", loss.item(), epoch)
        writer.add_scalar("Validation loss", loss_val.pow(2).mean(), epoch)

        pbar.update(1)

    pbar.update(1)
    pbar.close()
    return nn_approximator, np.array(loss_values)

def f_nn(nn: NN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model
    Internally calling the forward method when calling the class as a function"""
    return nn(x, t)

def get_current_time(fmt="%H:%M") -> str:
    tz = pytz.timezone('Europe/Berlin')
    now = datetime.datetime.now(tz)
    return now.strftime(fmt)
