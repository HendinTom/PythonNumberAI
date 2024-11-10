import os
import numpy as np
from PIL import Image
import struct

# Function to read the MNIST images from the raw binary file
def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Reading the header
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the images as a numpy array
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Path to the MNIST raw folder
raw_data_path = "data/MNIST/raw/"

# Read the training images from the raw MNIST files
train_images_path = os.path.join(raw_data_path, "train-images-idx3-ubyte")
images = read_mnist_images(train_images_path)

# Now, let's save the first 100 images as PNG or JPEG files
num_images_to_save = 100
for i in range(num_images_to_save):
    # Get the 28x28 image array
    image = images[i]

    # Convert the numpy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Save the image as PNG (you can also save as JPG by changing the format)
    pil_image.save(f"mnist_image_{i + 1}.png", format="PNG")  # Save as PNG
    # pil_image.save(f"mnist_image_{i + 1}.jpg", format="JPEG")  # Or as JPG