import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Define paths
path = Path("C:\\Users\\iliya\\Desktop\\kidney")
train_dir = "C:\\Users\\iliya\\Desktop\\kidney\\new kidney\\train"
val_dir = "C:\\Users\\iliya\\Desktop\\kidney\\new kidney\\val"


# Define image dimensions
img_height = 224
img_width = 224
batch_size = 32
num_epochs = 10
num_classes = 2

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest')

# Data augmentation for validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data generators for train and validation sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Load InceptionV3 base model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine base model and custom top layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    batch_size=batch_size,
    steps_per_epoch=train_generator.samples//batch_size,
    validation_steps=val_generator.samples//batch_size)

# Save the model
model.save('inception.h5')

# Load the saved model for testing
from tensorflow import keras
model = keras.models.load_model('inception.h5')

# Load and preprocess test image
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the user-provided image
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
# Make predictions
preds = model.predict(img)
print(preds)

# Define threshold for classification
threshold = 0.5
if preds[0][0] > threshold:
    print("The input image contains a kidney stone.")
else:
    print("The input image contains a normal kidney.")
