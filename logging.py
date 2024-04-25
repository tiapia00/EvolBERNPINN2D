def create_folder_date(directory, folder_name):
    folder_path = os.path.join(directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def get_last_modified_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    if not files:
        return None  # Return None if the folder is empty

    # Initialize variables to keep track of the most recently modified file
    last_modified_file = None
    last_modified_time = 0

    # Iterate over each file in the folder
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Check if the current path is a file (not a directory)
        if os.path.isfile(file_path):
            # Get the modification time of the file
            file_modified_time = os.path.getmtime(file_path)

            # Compare with the last modified time found
            if file_modified_time > last_modified_time:
                last_modified_time = file_modified_time
                last_modified_file = file_path

def get_current_time():
    current_time_utc = datetime.datetime.utcnow()
    target_timezone = pytz.timezone('Europe/Paris')
    return current_time_utc.astimezone(target_timezone)