import os
import datetime
import pytz

def pass_folder():
    date = get_current_time(fmt='%m-%d')
    time = get_current_time(fmt='%H:%M')
    direct = f'model/{date}/{time}'
    if not os.path.exists(direct):
        os.makedirs(direct)
        print(f"Folder '{direct}' created successfully")
    else:
        print(f"Folder '{direct}' already exists")
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