import os

folder_path = '/media/huifang/data/experiment/pix2pix/images/binary-square-without-transformer'  # Specify the folder path

# Get the list of files in the folder
files = os.listdir(folder_path)

# Iterate over the files
for filename in files:
    # Split the file name into number1, number2, and image name
    parts = filename.split('_')

    if len(parts) >= 3:
        # Construct the new file name without the first number
        new_filename = '_'.join(parts[1:])

        # Create the new file path
        new_filepath = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(os.path.join(folder_path, filename), new_filepath)
        print(f'Renamed {filename} to {new_filename}')

print('File renaming completed.')