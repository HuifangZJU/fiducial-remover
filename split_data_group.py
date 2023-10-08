import os
import matplotlib.pyplot as plt
badfile=[109,113,114,116,119,129,131,132,136,137,138,144,152,153,154,160,161,162,163,164,165]
def update_image_txt(image_txt_path, temp_folder_path,updated_txt_path):
    # Read the lines from image.txt
    with open(image_txt_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    data_level = os.listdir(temp_folder_path)
    for level in data_level:
        groupid = os.listdir(temp_folder_path+'/'+level)
        for id in groupid:
            images = os.listdir(temp_folder_path+'/'+level+'/'+id)
            for image in images:
                line_num = image.split('.')[0]
                line_num = int(line_num)
                if line_num in badfile:
                    continue
                image_path = lines[line_num].rstrip('\n')
                updated_line = f"{image_path} {level} {id}\n"
                updated_lines.append(updated_line)



    # for i, line in enumerate(lines):
    #     image_path = line.strip()
    #     data_level = ""
    #     group_id = ""
    #
    #     # Determine the data level based on the parent folder of the image
    #     parent_folder = os.path.dirname(temp_folder_path)
    #     if parent_folder.endswith('easy'):
    #         data_level = 'easy'
    #     elif parent_folder.endswith('moderate'):
    #         data_level = 'moderate'
    #     elif parent_folder.endswith('hard'):
    #         data_level = 'hard'
    #
    #     # Determine the group ID based on the sub-sub folder containing the image
    #     group_folder = os.path.basename(parent_folder)
    #     group_id = group_folder
    #
    #     # Update the line in image.txt with the path, data level, and group ID


    # Write the updated lines back to image.txt
    with open(updated_txt_path, 'w') as file:
        file.writelines(updated_lines)

image_txt_path = '/home/huifang/workspace/data/imagelists/st_image_trainable_fiducial.txt'
updated_txt_path = 'updated.txt'
temp_folder_path = './temp'
update_image_txt(image_txt_path, temp_folder_path,updated_txt_path)