import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from PIL import Image
from skimage.exposure import equalize_adapthist

# Folder paths
input_folder = "images"  # Replace with the path to your input folder
output_folder = "vessel_folder"  # Replace with the desired output folder path

# Step 1: Iterate through images in the input folder
for image_name in os.listdir(input_folder):
    # Read and preprocess the image
    image_path = os.path.join(input_folder, image_name)
    input_image = Image.open(image_path).convert("L")  # Read the image in grayscale

    # Preserve aspect ratio and resize image to fit within a 720x720 bounding box
    resized_image = input_image.resize((720, 720), Image.BICUBIC)

    # Enhance image quality using adaptive histogram equalization
    enhanced_image = equalize_adapthist(np.array(resized_image))

    # Step 2: Vessel Segmentation (Frangi vesselness filter)
    vesselness = frangi(enhanced_image)

    # Step 3: Post-processing
    thresholded_image = np.zeros_like(vesselness, dtype=np.uint8)
    thresholded_image[vesselness > 0.02] = 255

    # Step 4: Circular Mask
    center = (resized_image.size[0] // 2, resized_image.size[1] // 2)  # Calculate the center of the image
    radius = min(center[0], center[1]) - 10  # Define the radius of the circular mask

    # Create the circular mask
    mask = np.zeros_like(resized_image, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply the circular mask to the thresholded image
    masked_image = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)

    # Step 5: Save the extracted vessels  
    output_image_name = os.path.splitext(image_name)[0] + ".tif"
    output_image_path = os.path.join(output_folder, output_image_name)
    cv2.imwrite(output_image_path, masked_image)

    print(f"Vessels extracted and saved: {output_image_path}")
