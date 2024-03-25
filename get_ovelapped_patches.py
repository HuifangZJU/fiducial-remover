import json
import os
import cv2

def process_annotations(src_folder,img_folder, dest_folder, json_range):
    for i in range(json_range):
        json_path = os.path.join(src_folder, f"{i}.json")
        if not os.path.exists(json_path):
            continue
        print(str(i))
        with open(json_path, 'r') as file:
            data = json.load(file)
        image_path = os.path.join(img_folder, data['imagePath'])
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)

        for j, shape in enumerate(data['shapes']):
            if shape['label'] == 'hough_bad' and shape['shape_type'] == 'rectangle':
                points = shape['points']
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                rect = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
                h, w = rect.shape[:2]
                aspect_ratio = max(h, w) / min(h, w)

                if aspect_ratio > 1.5:
                    # Divide the rectangle into smaller patches
                    patches = divide_into_patches(rect, aspect_ratio)
                    for k, patch in enumerate(patches):
                        patch_path = os.path.join(dest_folder, f"{i}_{j}_{k}_orig.png")
                        cv2.imwrite(patch_path, patch)
                else:
                    patch_path = os.path.join(dest_folder, f"{i}_{j}_orig.png")
                    cv2.imwrite(patch_path, rect)


def divide_into_patches(rect, aspect_ratio):
    h, w = rect.shape[:2]
    patches = []

    if w > h:  # Width is greater than height
        n_patches = int(aspect_ratio // 1.5) + 1
        patch_width = w // n_patches
        for i in range(n_patches):
            patch = rect[:, i*patch_width:(i+1)*patch_width]
            patches.append(patch)
    else:  # Height is greater than width
        n_patches = int(aspect_ratio // 1.5) + 1
        patch_height = h // n_patches
        for i in range(n_patches):
            patch = rect[i*patch_height:(i+1)*patch_height, :]
            patches.append(patch)

    return patches

# Example usage
src_folder = '/home/huifang/workspace/code/fiducial_remover/temp_result/method/hough_detection/'  # Update this path
img_folder= '/home/huifang/workspace/code/fiducial_remover/location_annotation/'
dest_folder = '/home/huifang/workspace/code/fiducial_remover/temp_result/method/comparison/hough_bad/'  # Update this path
json_range = 167  # Number of json files

process_annotations(src_folder,img_folder,dest_folder, json_range)
