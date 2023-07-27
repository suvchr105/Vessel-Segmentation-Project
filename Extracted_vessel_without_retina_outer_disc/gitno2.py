import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi

# Step 1: Image Preprocessing
input_image = cv2.imread("healthy.jpg", 0)  # Read the image in grayscale
resized_image = cv2.resize(input_image, (256, 256))  # Resize image to a suitable resolution
normalized_image = resized_image / 255.0  # Normalize pixel values between 0 and 1

# Step 2: Vessel Segmentation (Frangi vesselness filter)
vesselness = frangi(normalized_image)

# Step 3: Post-processing
thresholded_image = np.zeros_like(vesselness, dtype=np.uint8)
thresholded_image[vesselness > 0.02] = 255

# Step 4: Circular Mask
center = (resized_image.shape[1] // 2, resized_image.shape[0] // 2)  # Calculate the center of the image
radius = min(center[0], center[1]) - 10  # Define the radius of the circular mask

# Create the circular mask
mask = np.zeros_like(resized_image, dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Apply the circular mask to the thresholded image
masked_image = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)

# Step 5: Visualization
plt.imshow(masked_image, cmap='gray')  # Display the extracted vessels
plt.title("Extracted Vessels (Central Region)")
plt.axis("off")
plt.show()