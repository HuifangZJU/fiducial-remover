import glob

# Define the root directory and pattern
root_dir = '/media/huifang/data/fiducial/original_data/10x/CytAssist/'  # Replace with your root directory path
# pattern = '*/spatial/cytassist_image.tiff'
pattern = '*/spatial/cytassist_image_cropped.png'
pattern2 = '*/spatial/tissue_hires_image.png'
# Use glob to find all files matching the pattern
image_paths = glob.glob(f'{root_dir}/{pattern}')
tissue_paths = glob.glob(f'{root_dir}/{pattern2}')
# Define the path for the output text file
output_file_path = '/home/huifang/workspace/data/imagelists/st_cytassist.txt'

# Write the image paths to the output file
with open(output_file_path, 'w') as file:
    for path,path2 in zip(image_paths,tissue_paths):
        file.write(path + ' ' + path2 + '\n')

print(f"Image paths saved to {output_file_path}")
