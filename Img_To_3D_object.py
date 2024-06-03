from PIL import Image
import numpy as np
import os

def shift_image(img, depth_img, shift_amount=10):
    # Ensure base image has alpha
    img = img.convert("RGBA")
    data = np.array(img)
    
    # Ensure depth image is grayscale (for single value)
    depth_img = depth_img.convert("L")
    depth_data = np.array(depth_img)
    
    # Calculate deltas for shifting
    deltas = ((depth_data / 255.0) * float(shift_amount)).astype(int)
    
    # Create the transparent resulting image with the same shape as the original image
    shifted_data = np.zeros_like(data)
    height, width, _ = data.shape
    
    # Shift the image pixels based on depth data
    for y, row in enumerate(deltas):
        for x, dx in enumerate(row):
            if x + dx < width and x + dx >= 0:
                shifted_data[y, x + dx] = data[y, x]
    
    # Convert the pixel data to an image
    shifted_image = Image.fromarray(shifted_data.astype(np.uint8))
    return shifted_image

# Load the images
img = Image.open("C://MScITPracs//OpenCV//OpenCV//CV//cube1.jpeg")
depth_img = Image.open("cube2.jpeg")

# Shift the image based on depth image
shifted_img = shift_image(img, depth_img, shift_amount=10)

# Show the shifted image
shifted_img.show()
