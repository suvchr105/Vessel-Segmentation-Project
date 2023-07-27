import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Load the dataset from the CSV file
dataset = pd.read_csv('dataset.csv')

# Load and preprocess the images and masks
images = []
masks = []

for index, row in dataset.iterrows():
    image = cv2.imread(row['ImageName'])
    mask = cv2.imread(row['MaskName'], cv2.IMREAD_GRAYSCALE)

    # Preprocess the images and masks
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(mask, axis=-1)

    image = img_to_array(image)
    mask = img_to_array(mask)

    images.append(image)
    masks.append(mask)

# Convert the image and mask lists to numpy arrays
images = np.array(images, dtype=np.float32) / 255.0
masks = np.array(masks, dtype=np.float32) / 255.0

# Split the dataset into train and test sets
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# Build the U-Net model
model = Sequential()
# Encoder
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
# Decoder
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling2D(size=2))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling2D(size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling2D(size=2))
model.add(Conv2D(1, kernel_size=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Define early stopping callback
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

# Train the model
model.fit(train_images, train_masks, batch_size=16, epochs=20, validation_data=(test_images, test_masks), callbacks=[early_stopping])

# Evaluate the model on test data
test_loss = model.evaluate(test_images, test_masks)
print('Test loss:', test_loss)

# Predict on test data
predictions = model.predict(test_images)

# Threshold the predictions to get binary masks
predictions_binary = (predictions > 0.5).astype(np.float32)

## Threshold the masks and predictions to get binary values
threshold = 0.5
test_masks_binary = (test_masks > threshold).astype(np.uint8)
predictions_binary = (predictions > threshold).astype(np.uint8)

# Flatten the masks and predictions
test_masks_flattened = test_masks_binary.reshape(-1)
predictions_flattened = predictions_binary.reshape(-1)

# Calculate accuracy, recall, and F1 score
accuracy = accuracy_score(test_masks_flattened, predictions_flattened)
recall = recall_score(test_masks_flattened, predictions_flattened)
f1 = f1_score(test_masks_flattened, predictions_flattened)

print('Accuracy:', accuracy)
print('Recall:', recall)
print('F1 Score:', f1)
