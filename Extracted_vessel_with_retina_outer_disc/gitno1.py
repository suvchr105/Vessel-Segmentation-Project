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
thresholded_image = np.zeros_like(vesselness)
thresholded_image[vesselness > 0.02] = 1

# Step 4: Visualization
plt.imshow(thresholded_image, cmap='gray')  # Display the extracted vessels
plt.title("Extracted Vessels")
plt.axis("off")
plt.show()
