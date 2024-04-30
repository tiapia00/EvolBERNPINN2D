import os
import datetime
import pytz
import yaml
import json
from string import Template

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
    with open(filename, 'r') as file:
        template_dict = json.load(file)

    parameters = template_dict['parameters']

    template_dict['pinn'] = {
        "x_end": parameters['x_end'],
        "y_end": parameters['y_end'],
        "t_end": parameters['t_end'],
        "n": parameters['n'],
        "hid_layers": parameters['hid_layers'],
        "neurons_per_layer": parameters['neurons_per_layer'],
        "lr": 0.002,
        "epoch": 50000,
        "weight_in": 1,
        "weight_bound": 1
    }

    template_dict['nn'] = {
        "x_end": parameters['x_end'],
        "t_end": parameters['t_end'],
        "n": parameters['n'],
        "hid_layers": parameters['hid_layers'],
        "neurons_per_layer": parameters['neurons_per_layer'],
        "lr": 0.002,
        "epoch": 8000
    }
    resolved_json_str = json.dumps(template_dict, indent=2)

    output_file_path = 'par_resolved.json'

    with open(output_file_path, 'w') as output_file:
        output_file.write(resolved_json_str)

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
        