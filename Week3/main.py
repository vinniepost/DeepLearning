from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

log_dir = "./logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping and normalizing the data
# X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
# X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)


# Definieer de ImageDataGenerator voor data-augmentatie.
# Gebruik ImageDataGenerator van Keras om verschillende
# transformaties toe te passen op de afbeeldingen tijdens het
# trainen:
# • Willekeurige rotatie tussen 0 en 10 graden
# • Willekeurige verschuiving horizontaal
# • Willekeurige verschuiving verticaal
# • Willekeurige zoom tussen 0 en 10%


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

augmented_data_gen = datagen.flow(X_train.reshape(-1, 28, 28, 1),y_train, batch_size=64)



model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Changed to categorical_crossentropy
              metrics=['accuracy'])


model.fit(augmented_data_gen, epochs=2, validation_data=(X_test.reshape(-1, 28, 28, 1), to_categorical(y_test, 10)), callbacks=[tensorboard_callback])

# model.fit(X_train, y_train, epochs=5, validation_split=0, callbacks=[tensorboard_callback])
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), to_categorical(y_test, 10), verbose=2)  # Ensure test labels are one-hot encoded

print(f'\nTest accuracy: {test_acc}')

import matplotlib.pyplot as plt

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
predictions = model.predict(X_test)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f'Predicted: {predicted_label}, True: {true_label}', color=color)

plt.savefig('prediction_plot.png')  # Save the plot as an image file
