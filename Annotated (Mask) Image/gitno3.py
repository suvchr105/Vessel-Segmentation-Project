import cv2
import glob
import os

# Folder path containing the TIFF images
folder_path = "images"  # Replace with the path to the folder containing the TIFF images

# Create a folder to save the resized mask images
mask_folder_path = os.path.join(folder_path, "mask_folder")
os.makedirs(mask_folder_path, exist_ok=True)

# Get a list of all TIFF image file paths in the folder
image_paths = glob.glob(os.path.join(folder_path, "*.tif"))

# Process each image in the folder
for image_path in image_paths:
    # Load the RGB retina image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create the mask
    _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize the mask image to the desired resolution
    resized_mask = cv2.resize(mask, (720, 720))

    # Save the resized mask image to the mask folder with the desired filename
    filename = os.path.splitext(os.path.basename(image_path))[0]  # Extract the filename without extension
    mask_image_path = os.path.join(mask_folder_path, f"{filename}_mask.tif")
    cv2.imwrite(mask_image_path, resized_mask)

    print(f"Processed: {image_path}")

print("All images processed successfully.")