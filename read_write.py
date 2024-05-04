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

def get_last_modified_file(folder_path, file_extension):
    try:
        # List to store (file_path, modification_time) tuples
        files = []

        # Walk through the directory tree (including subfolders)
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(file_extension):
                    file_path = os.path.join(root, filename)
                    modification_time = os.path.getmtime(file_path)
                    files.append((file_path, modification_time))

        # Sort files based on modification time (most recent first)
        files.sort(key=lambda x: x[1], reverse=True)

        if files:
            return files[0][0]  # Return the path of the most recently modified file
        else:
            print(f"No files with extension '{file_extension}' found in {folder_path} or its subfolders.")
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