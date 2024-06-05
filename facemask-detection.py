# Import required libraries.
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten ,Dense ,Dropout ,BatchNormalization
from tensorflow.keras.optimizers import Adamax
from sklearn.model_selection import train_test_split


from tensorflow.keras.utils import plot_model #for model visualization

# Load the dataset.

data_dir = "/content/sample_data/data/"


with_mask_dir = os.path.join(data_dir, "with_mask")
without_mask_dir = os.path.join(data_dir, "without_mask")

with_mask = os.listdir(with_mask_dir)[:5]
without_mask = os.listdir(without_mask_dir)[:5]

type(with_mask)

"""### Visulize first 5 img from both with and without mask datasets"""

# Plot the first 5 images from the "with_mask" class
plt.figure(figsize=(15, 6))
for i, file in enumerate(with_mask, 1):
    img_path = os.path.join(with_mask_dir, file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV uses BGR)
    plt.subplot(2, 5, i)
    plt.imshow(img)
    plt.axis('off')
    plt.title('With Mask')

# Plot the first 5 images from the "without_mask" class
for i, file in enumerate(without_mask, 1):
    img_path = os.path.join(without_mask_dir, file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV uses BGR)
    plt.subplot(2, 5, i + 5)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Without Mask')

plt.show()

"""### data preprocessing."""

data_with_mask= os.listdir(with_mask_dir)
data_without_mask= os.listdir(without_mask_dir)

len(data_with_mask)

len(data_without_mask)

"""##### We can see that we have balaced dataset here. -  no need for sampling

### annotate the data with labels
"""

## Do lebleing for withmask - 1, withoutmast -0.

with_mask_labels = [1]*len(data_with_mask)
print(f"With Mask labels" ,with_mask_labels[0:10])

without_mask_labels = [0]*len(data_without_mask)
print(f"Without Mask Labels", without_mask_labels[0:10])

Labels = with_mask_labels + without_mask_labels
print(f"Labels", Labels[0:10])
print(f"Labels", Labels[-10:])

"""### Resize images and convert to numpy array"""

#resize images and convert to numpy array
import os
from PIL import Image

images = []
def load_images_from_folder(folder):

    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        img = img.resize((128,128))
        img = img.convert('RGB')
        images.append(np.array(img))
    return images

with_mask_images = load_images_from_folder('/content/sample_data/data/with_mask')
without_mask_images = load_images_from_folder('/content/sample_data/data/without_mask')

len(images)

images[0].shape

images[100].shape

X = np.array(images)
y = np.array(Labels)

X.shape

y.shape

"""### Train Test Split"""

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

"""### Scalle the data.

"""

X_train = X_train/255

X_test = X_test/255

# Save each model performance

model_dict ={}

print(model_dict)

"""### Building a Model with ResNet152V2"""

image_size = (128,128)
channels = 3
image_shape = (image_size[0],image_size[1], channels)

# 1> ResNet152V2
# base_model = tf.keras.applications.ResNet152V2(
#     include_top=False,
#     weights="imagenet",
#      input_shape=image_shape)

# 2> Xception
# base_model = tf.keras.applications.Xception(
#     include_top=False,
#     weights="imagenet",
#     input_shape=image_shape

# 3> VGG16
# base_model = tf.keras.applications.VGG16(
#     include_top=False,
#     weights="imagenet",
#     input_shape=image_shape
#  )

# 4> MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=image_shape
)

# 5> InceptionV3
# base_model = tf.keras.applications.InceptionV3(
#     include_top=False,
#     weights="imagenet",
#     input_shape=image_shape
# )

# 6> DenseNet121
# base_model = tf.keras.applications.DenseNet121(
#     include_top=False,
#     weights="imagenet",
#     input_shape=image_shape
# )

print(base_model)

model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),
    Dense(256, activation ='relu'),
    Dropout(rate=0.2),
    BatchNormalization(),
    Dense(128, activation ='relu'),
    Dropout(rate=0.2),
    BatchNormalization(),
    Dense(64, activation ='relu'),
    Dropout(rate=0.2),

    Dense(2, activation ='sigmoid')

])

# Compile the model.

model.compile(Adamax(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

""" Total params: 58,331,648 (222.52 MB)

 Trainable params: 58,187,904 (221.97 MB)

 Non-trainable params: 143,744 (561.50 KB)
"""

import time
from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

start_time = time.time()

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=25, validation_split=0.1, callbacks=[early_stopping])

# Record the end time
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time

# Get the original model name
model_name = model.name if model.name != 'sequential_7' else model.layers[0].name

print(model_name)

# Get the final training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

# Get the final training and validation loss
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

# Store the model information in the global dictionary
model_dict[model_name] = {
        'Model Name': model_name,
        'Training Time (seconds)': training_time,
        'Train Accuracy': train_accuracy,
        'Validation Accuracy': val_accuracy,
        'Train Loss': train_loss,
        'Validation Loss': val_loss
    }

print(model_dict[model_name])

print(model_dict)

import matplotlib.pyplot as plt

# Extract model names
model_names = list(model_dict.keys())

# Extract model metrics
train_accuracies = [model_dict[model_name]['Train Accuracy'] for model_name in model_names]
val_accuracies = [model_dict[model_name]['Validation Accuracy'] for model_name in model_names]
training_times = [model_dict[model_name]['Training Time (seconds)'] for model_name in model_names]

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.bar(model_names, train_accuracies, color='b', label='Train Accuracy')
plt.bar(model_names, val_accuracies, color='r', alpha=0.5, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""### We can see all these model gave almost 99 percent accuracy on both train and test dataset.

### However we should also look for time taken by each model.
"""

# Plot training time
plt.figure(figsize=(10, 5))
plt.bar(model_names, training_times, color='g')
plt.title('Training Time Comparison')
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Extract model metrics
train_losses = [model_dict[model_name]['Train Loss'] for model_name in model_names]
val_losses = [model_dict[model_name]['Validation Loss'] for model_name in model_names]

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(model_names, train_losses, marker='o', color='b', label='Train Loss')
plt.plot(model_names, val_losses, marker='o', color='r', label='Validation Loss')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""

# choosing the appropriate model for different scenarios:

1. **High Accuracy with Moderate Training Time**:
   - For applications where high accuracy is crucial and there are no strict constraints on training time, models like 'inception_v3' and 'densenet121' can be suitable choices. These models achieve high validation accuracy (close to 1.0) with reasonable training times (around 85-125 seconds). InceptionV3 captures features at multiple scales, while DenseNet121 enhances feature reuse, leading to high accuracy.

2. **Faster Training with Slightly Lower Accuracy**:
   - If you prioritize faster training times while still maintaining relatively high accuracy, models like 'xception' and 'mobilenetv2_1.00_128' offer good performance. These models achieve validation accuracies above 0.97 with training times ranging from 60 to 102 seconds. Xception's depthwise separable convolutions and MobileNetV2's lightweight architectures make them efficient choices for faster training.

3. **Balanced Performance with Lower Complexity**:
   - For scenarios where a balance between performance and model complexity is desired, models like 'resnet152v2' and 'vgg16' can be considered. These models achieve decent validation accuracies (around 0.91 to 0.99) with lower training times compared to some of the more complex models. ResNet152V2's residual connections enable training of deeper networks, while VGG16's simple and uniform architecture makes it easy to understand and implement.

4. **Resource-Constrained Environments**:
   - In resource-constrained environments where computational resources are limited, models like 'mobilenetv2_1.00_128' and 'vgg16' may be preferable due to their lower model complexity and faster training times. MobileNetV2's lightweight architectures and VGG16's simplicity make them efficient choices for such environments.

5. **High Performance with High Complexity**:
   - If computational resources are not a concern and achieving the highest possible accuracy is the primary goal, models like 'densenet121' and 'inception_v3' offer excellent performance with validation accuracies close to 1.0. However, these models come with higher complexity and longer training times. DenseNet121's dense connections and InceptionV3's multiscale feature extraction capabilities contribute to their high performance.

Ultimately, the choice of model depends on the specific requirements of your application, including the trade-off between accuracy, training time, and model complexity, as well as the available computational resources.
"""

#Classification Report

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))

# Do some prediction

input_image_path = '/content/sample_data/data/with_mask/with_mask_1024.jpg'

input_image = cv2.imread(input_image_path)

plt.imshow(input_image)
plt.show()

input_image_resized = cv2.resize(input_image, (128,128))

input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

input_prediction = model.predict(input_image_reshaped)

print(input_prediction)


input_pred_label = np.argmax(input_prediction)

print(input_pred_label)


if input_pred_label == 1:

  print('The person in the image is wearing a mask')

else:

  print('The person in the image is not wearing a mask')

# Save the model
model.save('/content/sample_data/face_mask_detector_model.h5')

from tensorflow.keras.models import load_model

# Load the model
model = load_model('/content/sample_data/face_mask_detector_model.h5')

!pip install mtcnn

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the trained model
model = load_model('/content/sample_data/face_mask_detector_model.h5')

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        # Get the bounding box coordinates
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height

        # Extract the face region
        face_region = frame[y:y2, x:x2]

        # Preprocess the face region
        face_resized = cv2.resize(face_region, (128, 128))  # Resize to match the input shape of the model
        face_normalized = face_resized / 255.0              # Normalize the pixel values
        face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))  # Reshape for the model input

        # Predict mask or no mask
        prediction = model.predict(face_reshaped)
        (mask, without_mask) = prediction[0]

        # Determine the class label and color
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
