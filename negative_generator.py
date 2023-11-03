import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Declare global variables
coordinates = []
selecting = False
start_point = None  # Variable to store the starting point of the rectangle
ax = None
image = None


# Create a callback function for mouse events
def select_region(event):
    global coordinates, selecting, start_point, ax, image
    if event.button == 1 and event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        if not selecting:
            # Start of selection (left upper corner)
            start_point = (x, y)
            selecting = True
        else:
            # End of selection (right lower corner)
            end_point = (x, y)
            coordinates.append((start_point, end_point))
            print(f"Selected coordinates: ({start_point[0]}, {start_point[1]}) to ({end_point[0]}, {end_point[1]})")
            selecting = False
            ax.add_patch(Rectangle(start_point, end_point[0] - start_point[0], end_point[1] - start_point[1],
                                   linewidth=2, edgecolor='g', facecolor='none'))
            plt.draw()
    elif event.button == 3:  # Right-click to clear selection
        coordinates = []
        selecting = False
        ax.clear()
        ax.imshow(image)
        ax.set_title(
            "Select a rectangle by left-clicking the left upper corner and then the right lower corner, right-click to clear, and press 'q' to save and quit.")
        plt.draw()
        print("Cleared selection")


# Read a list of image file paths with additional information
with open('/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt', 'r') as file:
    lines = file.read().splitlines()

for line in lines:
    elements = line.split()
    if len(elements) < 1:
        continue  # Skip lines without an image path
    image_path = elements[0]

    # Load the current image
    image = plt.imread(image_path)
    coordinates = []
    selecting = False

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(image)
    ax.set_title(
        "Select a rectangle by left-clicking the left upper corner and then the right lower corner, right-click to clear, and press 'q' to save and quit.")

    fig.canvas.mpl_connect('button_press_event', select_region)

    while True:
        # plt.pause(0.01)
        if plt.waitforbuttonpress(0.01):
            break

    if len(coordinates) > 0:
        # Save the selected coordinates to a file or process them as needed
        with open('coordinates.txt', 'a') as file:
            file.write(f"{image_path} ")
            for (x1, y1), (x2, y2) in coordinates:
                file.write(f"{x1} {y1} {x2} {y2}")
            file.write('\n')
        print(f"Coordinates for {image_path} saved.")

    plt.close()
