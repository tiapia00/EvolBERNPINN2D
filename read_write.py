import os
import datetime
import pytz
import yaml
import json
from jinja2 import Template

def pass_folder():
    date = get_current_time(fmt='%m-%d')
    time = get_current_time(fmt='%H:%M')
    direct = f'/model/{date}/{time}'
    if not os.path.exists(direct):
        os.makedirs(direct)
        print(f"Folder '{folder_name}' created successfully at '{os.path.abspath(path)}'.")
    else:
        print(f"Folder '{folder_name}' already exists at '{os.path.abspath(path)}'.")
    return direct

def create_folder_date(directory, folder_name):
    folder_path = os.path.join(directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def get_last_modified_file(folder_path):
    # Ensure the provided path is a directory
    try:
        # Get a list of all files in the specified directory
        files = os.listdir(folder_path)

        # Filter out directories and get a list of (file, modified_time) tuples
        files = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files if os.path.isfile(os.path.join(folder_path, f))]

        # Sort files based on modified time (second element of the tuple)
        files.sort(key=lambda x: x[1], reverse=True)

        if files:
            # Return the file with the latest modified time
            return os.path.join(folder_path, files[0][0])
        else:
            print(f"No files found in {folder_path}.")
            return None

    except OSError as e:
        print(f"Error: {e}")
        return None

def get_current_time(timezone_name='Europe/Paris', fmt='%Y-%m-%d %H:%M:%S'):
    current_time_utc = datetime.datetime.utcnow()
    target_timezone = pytz.timezone(timezone_name)
    current_time_local = current_time_utc.astimezone(target_timezone)
    time_str = current_time_local.strftime(fmt)
    return time_str

def resolve_json(filename):
    
    parameters = {
    "x_end": 1,
    "y_end": 1,
    "t_end": 1,
    "n": 40,
    "hid_layers": 3,
    "neurons_per_layer": 40
}
    with open(filename, 'r') as f:
        template_content = f.read()
    
    template = Template(template_content)

    rendered_json = template.render(parameters=parameters)

    with open('par_resolved.json', 'w') as f:
        f.write(rendered_json)

def get_params_PINN(filename):
    
    with open(filename, 'r') as file:
        config = json.load(file)
        
    pinn = config['pinn']
    
    Lx = pinn['x_end']
    Ly = pinn['y_end']
    T = pinn['t_end']
    n = pinn['n']
    hid_layers = pinn['hid_layers']
    dim_hidden = pinn['neurons_per_layer']
    lr = pinn['lr']
    epoch = pinn['epoch']
    weight_in = pinn['weight_in']
    weight_bound = pinn['weight_bound']
    
    return Lx, Ly, T, n, hid_layers, dim_hidden, lr, epoch, weight_in, weight_bound

def get_params_NN(filename):
    
    with open(filename, 'r') as file:
        config = json.load(file)
        
    nn = config['nn']
    
    Lx = nn['x_end']
    T = nn['t_end']
    n = nn['n']
    hid_layers = nn['hid_layers']
    dim_hidden = nn['neurons_per_layer']
    lr = nn['lr']
    epoch = nn['epoch']
    
    return Lx, T, n, hid_layers, dim_hidden, lr, epoch

def get_params_mat(filename):
    
    with open(filename, 'r') as file:
        config = json.load(file)
        
    mat_par = config['params_mat']
    
    rho = mat_par['rho']
    E = mat_par['E']
    nu = mat_par['nu']
    h = mat_par['h']
    
    return E, rho, h, nu
        