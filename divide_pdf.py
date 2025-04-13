from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import PyPDF2
from matplotlib.patches import Rectangle
import numpy as np

def select_area(ax, image):
    """Function to interactively select the crop area."""
    ax.imshow(image)

    print("Please draw a rectangle over the area to crop (click two corners):")
    rect = plt.ginput(2)
    print("Selected coordinates:", rect)
    # plt.show()
    rect_patch = Rectangle((min(rect[0][0], rect[1][0]), min(rect[0][1], rect[1][1])),
                           abs(rect[1][0] - rect[0][0]), abs(rect[1][1] - rect[0][1]),
                           linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect_patch)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
    return rect

def interactive_crop_pdf(input_path, output_folder):
    images = convert_from_path(input_path, dpi=300)  # Convert PDF to list of images
    dpi_scale = 72 / 200
    with open(input_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        assert len(images) == len(reader.pages), "Number of images and pages should match"

        for i, image in enumerate(images):
            # if i <6:
            #     continue
            # print(image.size)
            fig, ax = plt.subplots()
            # ax.imshow(image)
            points = select_area(ax, image)

            # Calculate crop box in PDF coordinates
            points = np.array(points)
            points[:, 1] = image.height - points[:, 1]  # Invert y-coordinate
            points *= dpi_scale  # Scale from image resolution to PDF points
            lower_left = points.min(axis=0)
            upper_right = points.max(axis=0)

            # Set crop box
            page = reader.pages[i]
            page.cropbox.lower_left = lower_left
            page.cropbox.upper_right = upper_right

            writer = PyPDF2.PdfWriter()
            writer.add_page(page)
            with open(f'{output_folder}/cropped_page_{i+1}.pdf', 'wb') as outfile:
                writer.write(outfile)


# Usage
interactive_crop_pdf("/home/huifang/workspace/paper/fiducial application/gene_imputation.pdf", "/home/huifang/workspace/paper/fiducial application/")
