import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tensorflow.keras.layers import Dropout

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
inputs = Input(shape=(256, 256, 3))
conv1 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv1)
conv2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=2)(conv2)
conv3 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=2)(conv3)
conv4 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=2)(drop4)
conv5 = Conv2D(1024, kernel_size=3, activation='relu', padding='same')(pool4)
conv5 = Conv2D(1024, kernel_size=3, activation='relu', padding='same')(conv5)
drop5 = Dropout(0.5)(conv5)
up6 = Conv2D(512, kernel_size=2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(merge6)
conv6 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(conv6)
up7 = Conv2D(256, kernel_size=2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(merge7)
conv7 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(conv7)
up8 = Conv2D(128, kernel_size=2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(merge8)
conv8 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv8)
up9 = Conv2D(64, kernel_size=2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(merge9)
conv9 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv9)
conv10 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=conv10)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

# Train the model
model.fit(train_images, train_masks, batch_size=8, epochs=10, validation_data=(test_images, test_masks), callbacks=[early_stopping])

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_masks)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Predict on test data
predictions = model.predict(test_images)

# Threshold the predictions to get binary masks
predictions_binary = (predictions > 0.5).astype(np.float32)

# Threshold the masks and predictions to get binary values
threshold = 0.5
test_masks_binary = (test_masks > threshold).astype(np.uint8)
predictions_binary = (predictions > threshold).astype(np.uint8)

# Flatten the masks and predictions
test_masks_flattened = test_masks_binary.reshape(-1)
predictions_flattened = predictions_binary.reshape(-1)

# Calculate accuracy, recall, precision, and F1 score
accuracy = accuracy_score(test_masks_flattened, predictions_flattened)
recall = recall_score(test_masks_flattened, predictions_flattened)
precision = precision_score(test_masks_flattened, predictions_flattened)
f1 = f1_score(test_masks_flattened, predictions_flattened)

print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)
