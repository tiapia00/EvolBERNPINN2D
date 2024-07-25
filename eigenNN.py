## Neural Network

#matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



# class for the neural network
class myPow2(nn.Module):
    def __init__(self):
        super().__init__()
        #self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, 2)
class myPow4(nn.Module):
    def __init__(self):
        super().__init__()
        #self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, 4)


class MLNet(nn.Module):
    # define activation function structure
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        #self.layer1 = nn.Linear(input_dim, hidden_dim)
        #self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        #self.layer3 = nn.Linear(hidden_dim, output_dim)


        self.hidden_dim = hidden_dim
        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            #nn.Sigmoid(),
            #nn.Tanh(),
            nn.ReLU(),
            #myPow2(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            #nn.Tanh(),
            #myPow2(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #myPow2(),
            #nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            #nn.Tanh()
            #myPow()
            #nn.ReLU()
        )


    # forward pass
    def forward(self, x):

        #state1 = (self.layer1(x))
        #state2 = (self.layer2(state1))**2
        #out = self.layer3(state2)
        out = self.fcnn1(x)
        return out


# evaluate model
def evaluate(model, input, output):
    with torch.no_grad():
        model.eval()  # set model to evaluation mode
        outputs = []  # create empty lists to store the results
        targets = []
        testlosses = []

        out = model(input.to(device))  # call the model, i.e. perform the actual inference

        #out = denormalize(out, ef_min, ef_max)


        outputs.append(out.cpu().detach().numpy())
        targets.append(output.cpu().detach().numpy())
        testlosses.append(criterion(out, output.to(device)).item())

    return outputs, targets, testlosses



# function to train model
def train(train_loader, learn_rate, epochs, vali, earlystopping):

    # instantiate the neural network
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device)  # and move it to the "device" (in case we use a gpu)
    # set optimizer, learning rate and L2-regularization
    optimizer = torch.optim.Adamax(model.parameters(), lr=learn_rate) #, amsgrad=True
    avg_losses_train = torch.zeros(epochs+1)
    losses_vali = torch.zeros(epochs+1)

    pbar = tqdm(total=epochs, desc="Training", position=0)

    for epoch in range(epochs+1):
        model.train()  # set the model into train mode
        avg_loss_train = 0.  # initializations
        counter = 0
        if epoch == 1950 :
            #if epoch % 5 ==0:
            #learn_rate = learn_rate/10
            learn_rate = learn_rate/10
            print('learnrate', learn_rate)
            optimizer = torch.optim.Adamax(model.parameters(), lr=learn_rate)

        for x, label in train_loader:
            counter += 1  # We count, how many batches we did within the current epoch
            model.zero_grad()  # reset gradients
            out_train = model(x.to(device))  # call model on current batch
            loss_train = criterion(out_train, label.to(device))  # calculate training loss
            loss_train.backward()  # backward pass
            optimizer.step()  # use optimizer to adjust model
            avg_loss_train += loss_train.item()  # save loss values

        out_vali = model(vali.to(device))  # call model on validation data
        loss_vali = criterion(out_vali, vali_lam.to(device))  # calculate validation loss
        losses_vali[epoch] = loss_vali.item()  # save validation loss
        avg_losses_train[epoch] = avg_loss_train / len(train_loader)  # save training loss

        pbar.set_description(f"Loss: {loss_train.item():.3e}")
        pbar.update(1)

    pbar.update(1)
    pbar.close()
    # plot loss curves
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(avg_losses_train[:epoch])
    ax2.plot(losses_vali[:epoch])
    #if earlystopping and best_epoch is not None:
        #print(f'Minimum validation loss: {best_loss_vali} in epoch {best_epoch}')
        #ax2.plot(best_epoch*np.ones(2), np.linspace(0, 1.5, 2), color='red')
    plt.title("Losses (MSE, reduction=mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss", "Stopping point (regularization)"])
    plt.grid(visible=True, which='both', axis='both')
    plt.savefig(Path(IMAGE_DIRECTORY, f"{'loss' + filename}.png"))
    print(f'normal Validation loss after {epoch} epochs: {losses_vali[epoch]}')

    # return model and losses
    return model, avg_losses_train, losses_vali

def normalize(value, min, max): #auf einen Wert zwischen 0 und 1 projizieren
    #norm_val = value
    #norm_val = (value - min) / (max - min)
    norm_val = value / max
    return norm_val

def denormalize(norm_val, min, max):    #get the orinal value
    #value = norm_val
    #value = norm_val * (max - min) + min
    value = norm_val * max
    return value

def normalizematr(valmatr, range):
    norm_valmatr = torch.zeros_like(valmatr)
    for count in np.arange(0, valmatr.size()[1]-1):
        norm_valmatr[:, count] = normalize(valmatr[:, count], range[0, count], range[1, count])
    return norm_valmatr

def denormalizematr(norm_valmatr, interval):
    valmatr = torch.zeros_like(norm_valmatr)
    for count in range(0, norm_valmatr.size()[1]-1):
        valmatr[:, count] = denormalize(norm_valmatr[:, count], interval[0, count], interval[1, count])
    return valmatr

def plot_output(data_beam, eigenfrequencies, ef):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Frequenz ' + str(ef))
    ax[0, 0].scatter(data_beam[:, 0], eigenfrequencies[:, ef], s=0.1)
    ax[0, 0].set_title('rho sortiert')
    ax[0, 1].scatter(data_beam[:, 1], eigenfrequencies[:, ef], s=0.1)
    ax[0, 1].set_title('E sortiert')
    ax[1, 0].scatter(data_beam[:, 2], eigenfrequencies[:, ef], s=0.1)
    ax[1, 0].set_title('l sortiert')
    ax[1, 1].scatter(data_beam[:, 3], eigenfrequencies[:, ef], s=0.1)
    ax[1, 1].set_title('b sortiert')
    plt.tight_layout()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)#64)
    torch.set_num_threads(10)
    device = torch.device("cpu")  # choose device
    print(f"Using {device} device")
    device = torch.device(device)

    CURRENT_FILE = Path(__file__)
    FILE_DIRECTORY = CURRENT_FILE.parent
    MODE_DIRECTORY = Path(FILE_DIRECTORY, "data/vergleichsmoden")
    FEM_DIRECTORY = Path(FILE_DIRECTORY, "data/fem")
    OUT_DIRECTORY = Path(FILE_DIRECTORY, "data/out")
    IMAGE_DIRECTORY = Path(FILE_DIRECTORY, "eigenestimator")

    samples = 31000
    rho_min = 1e3  # Alu ca. 2.7e3, steel ca. 7.85 kg/m^3   Wikipedia
    rho_max = 10e3
    E_min = 50e9  # Alu ca. 68 GPa, steel ca. 200GPa       Wikipedia
    E_max = 250e9
    l_min = 6  # min 6:1, max 40:1
    l_max = 20
    b_min = 0.5
    b_max = 1
    n_frequencies = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    earlystopping = False  # implement early stopping for regularization

    # hyperparameters
    batch_size = 100  # number of samples to be presented in one training step
    hidden_dim = 60  # number of neurons in hidden layers
    input_dim = 4
    output_dim = len(n_frequencies)

    epochs = 1000  # number of training epochs
    lr = 0.001  # learning rate

    criterion = nn.MSELoss(reduction="mean")  # loss criterion: mean square error

    # loading training data
    filename = '600_fem_180x10'
    [data_beam, data_range, eigenfrequencies] = torch.load(Path(FEM_DIRECTORY, filename))
    #data_beam[:, 2] = 5 * data_beam[:, 2]
    beam = data_beam
    anafrequencies = torch.zeros_like(eigenfrequencies)
    #biegefreuenzen
    anafrequencies[:, 0] = 1.87510407
    anafrequencies[:, 1] = 4.69409113
    anafrequencies[:, 2] = 7.85475744
    anafrequencies[:, 3] = 10.9955407
    anafrequencies[:, 4] = 14.1371684
    anafrequencies[:, 5] = 17.2787595
    anafrequencies = torch.pow(anafrequencies, 4)
    anafrequencies *= 1/12
    for i in range(6):
        anafrequencies[:, i] = anafrequencies[:, i] /beam[:, 0]
        anafrequencies[:, i] = anafrequencies[:, i] * beam[:, 1]
        anafrequencies[:, i] = anafrequencies[:, i] /torch.pow(beam[:, 2], 4)
        anafrequencies[:, i] = anafrequencies[:, i] * torch.pow(beam[:, 3], 2)

    #longditudinal
    #for i in range(3):
    anafrequencies[:, 6] =  torch.pow((torch.pi/beam[:, 2]), 2) * beam[:, 1]/beam[:, 0] * 0.5**2
    anafrequencies[:, 7] =  torch.pow((torch.pi/beam[:, 2]), 2) * beam[:, 1]/beam[:, 0] * 1.5**2
    anafrequencies[:, 8] =  torch.pow((torch.pi/beam[:, 2]), 2) * beam[:, 1]/beam[:, 0] * 2.5**2

    anafrequencies = torch.sqrt(anafrequencies)

    ## training  with analytical data
    #eigenfrequencies = anafrequencies


    filename = 'fem_180x10_NN_2'
    eigenfrequencies = eigenfrequencies[:, n_frequencies]

    ef_range = torch.zeros(2, len(n_frequencies))
    ef_range[0, :] = torch.min(eigenfrequencies, 0)[0]
    ef_range[1, :] = torch.max(eigenfrequencies, 0)[0]

    eigenfrequencies = normalizematr(eigenfrequencies, ef_range)

    #data
    data_beam = normalizematr(data_beam, data_range)

    # splitting data into train and vali set
    train_beam, vali_beam = torch.split(data_beam, [550, 50], dim=0)
    train_lam, vali_lam = torch.split(eigenfrequencies, [550, 50], dim=0)

    # creating dataset and trainloader
    train_data = TensorDataset(train_beam, train_lam)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)


    model, avg_losses_train, losses_vali, = train(train_loader, lr, epochs, vali_beam, earlystopping)
    torch.save(model.state_dict(), 'model//eigenestmodel.pth')
    torch.save(ef_range, 'model//ef_range.pt')
