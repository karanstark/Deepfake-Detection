import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paths
TRAIN_DIR = "train_images"  # Folder containing images
LABELS_FILE = "fake_cifake_preds.json"
IMAGE_EXTENSION = ".png"

# 1️⃣ Load labels
print("Loading labels and data paths...")
df_labels = pd.read_json(LABELS_FILE)

# Ensure filenames are strings
df_labels.index = df_labels.index.astype(str)

# Create paths for fake and real images
def get_image_path(row):
    if row['prediction'] == 'fake':
        return os.path.join(TRAIN_DIR, 'fake_cifake_images-20251019T154446Z-1-001', 'fake_cifake_images', f"{row.name}{IMAGE_EXTENSION}")
    else:
        return os.path.join(TRAIN_DIR, 'real_cifake_images-20251019T154723Z-1-001', 'real_cifake_images', f"{row.name}{IMAGE_EXTENSION}")

# Generate paths based on prediction
df_labels['path'] = df_labels.apply(get_image_path, axis=1)

print(f"✅ Loaded {len(df_labels)} image labels successfully.")

# 2️⃣ Image data generator (for data augmentation)
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Use prediction column directly as label
df_labels['label'] = df_labels['prediction']

# Create generators
train_gen = datagen.flow_from_dataframe(
    dataframe=df_labels,
    x_col="path",
    y_col="label",
    subset="training",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df_labels,
    x_col="path",
    y_col="label",
    subset="validation",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

# 4️⃣ Build CNN model
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5️⃣ Train model
print("Starting training...")
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# 6️⃣ Save model
model.save("deepfake_cnn_model.h5")
print("✅ Model training complete and saved as deepfake_cnn_model.h5")
