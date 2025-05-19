import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir))  # Get class names
    print(f"Found {len(classes)} classes: {classes}")

    for label in classes:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir): continue

        for img_file in os.listdir(label_dir)[:1000]:  # Optional: limit to 1000 images/class for speed
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load and preprocess
data_dir = r"C:\Users\vigne\Downloads\archive (4)\asl_alphabet_train\asl_alphabet_train"
X, y = load_data(data_dir)

# Normalize
X = X.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

# Encode labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Save the label map for inference
import pickle
with open("label_map.pkl", "wb") as f:
    pickle.dump(lb, f)

# Show class distribution
plt.figure(figsize=(16, 5))
sns.countplot(x=y, order=np.unique(y))
plt.title('Class Distribution')
plt.xlabel('Gesture Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(75, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(50, (3, 3), activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(25, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(lb.classes_), activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Augment and train
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                    epochs=30,
                    validation_data=(X_test, y_test))

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save model
model.save('asl_cnn_model2.keras')
print("Model saved as asl_cnn_model.keras")
