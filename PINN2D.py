import json
import numpy as np
import os
from pinn import *
from nn import NN
import numpy as np
from read_write import get_last_modified_file, pass_folder

retrain_init = False
retrain_PINN = True

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

get_ipython().run_line_magic('run', 'par.py')


if retrain_init:
    get_ipython().run_line_magic('run', 'initialization_NN.py')


# In[10]:


from read_write import get_params
from par import to_matpar_PINN

E, rho, _ , nu = get_params(mat_par)

lam, mu = to_matpar_PINN(E, nu)

Lx, Ly, T, n_train, layers, dim_hidden, lr, epochs, weight_IN, weight_BOUND = get_params(pinn_par)

x_domain = np.array([0.0, Lx])/Lx
y_domain = np.array([0.0, Ly])/Lx
t_domain = np.array([0.0, T])/T


# In[11]:


pinn = PINN(layers, dim_hidden, act=nn.Tanh()).to(device)

loss_fn = Loss(
    x_domain,
    y_domain,
    t_domain,
    n_train,
    return_adim(x_domain, t_domain, rho, mu, lam),
    initial_conditions,
    weight_IN,
    weight_BOUND
)


# In[12]:


path = pass_folder()

if retrain_PINN:
    pinn = PINN(layers, dim_hidden, act=nn.Tanh()).to(device)

    loss_fn = Loss(
        x_domain,
        y_domain,
        t_domain,
        n_train,
        return_adim(x_domain, t_domain, rho, mu, lam),
        initial_conditions,
        weight_IN,
        weight_BOUND
    )
    
    filename_model = get_last_modified_file('in_model')
    pretrained_model_dict = torch.load(filename_model, map_location=torch.device(device))

    pretrained_model = NN(layers, dim_hidden, 2, 1)
    pretrained_model.load_state_dict(pretrained_model_dict)

    for i in np.arange(len(pinn.middle_layers)):
        pinn_layer = pinn.middle_layers[i]
        pretrained_layer = pretrained_model.middle_layers[i]
        pinn.middle_layers[i].weight.data.copy_(pretrained_model.middle_layers[i].weight)
        pinn.middle_layers[i].bias.data.copy_(pretrained_model.middle_layers[i].bias)

    pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=lr, max_epochs=epochs, path=path)
    
    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(path, model_name)
    
    torch.save(pinn_trained.state_dict(), model_path)
    
else:
    pinn_trained = PINN(layers, dim_hidden, act=nn.Tanh()).to(device)
    filename = get_last_modified_file('model')
    pinn_trained.load_state_dict(torch.load(filename, map_location = device))


# # Train new network with pretrained weights and biases in middle layers

# In[ ]:


pinn_trained.eval()


# In[ ]:


from plots import plot_initial_conditions, plot_uy

x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_train)
t_value = 0.0
t = torch.full_like(x, t_value)
x = x.to(device)
y = y.to(device)
t = t.to(device)
z = f(pinn_trained, x ,y, t)
ux_0, uy_0 = initial_conditions(x, y, Lx, i = 1)
z_0 = torch.cat((ux_0, uy_0), dim=1)

plot_initial_conditions(z_0, y, x, 'Initial conditions - analytical', n_train, from_pinn = 0)
plot_initial_conditions(z, y, x, 'Initial conditions - NN', n_train)


# In[ ]:


x, y, t = get_interior_points(x_domain, y_domain, t_domain, n_train)
plot_uy(pinn_trained, x, y, t, n_train, path)


# # To be added
# - ~separate loss in more bars to see how the various loss term come to zero~
# - see if some quadrature rule has been implemented
# - scheme of weights initialization in order to automatically satisfy initial conditions
# - plots (in progress)
# - NN operators (to generalize results)
# - try to implement function that allows that satisfy initial conditions?
