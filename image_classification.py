import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(150, 150),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val",
    image_size=(150, 150),
    batch_size=32
)

# Normalize images
normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print("Accuracy:", accuracy)
loss, accuracy = model.evaluate(test_data)
print("Final Test Accuracy:", accuracy)