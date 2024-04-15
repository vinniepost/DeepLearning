import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import keras_tuner as kt

# Generate moons dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the input data for RNN and LSTM
X_train_rnn = X_train.reshape(-1, 1, 2)
X_test_rnn = X_test.reshape(-1, 1, 2)

def build_model(hp):
    model = Sequential()
    model_type = hp.Choice('model_type', ['RNN', 'LSTM'])
    for i in range(hp.Int('n_layers', 1, 3)):
        if model_type == 'RNN':
            model.add(SimpleRNN(units=hp.Int('units', min_value=10, max_value=50, step=10),
                                return_sequences=i < hp.get('n_layers') - 1,
                                input_shape=[1, 2]))
        elif model_type == 'LSTM':
            model.add(LSTM(units=hp.Int('units', min_value=10, max_value=50, step=10),
                           return_sequences=i < hp.get('n_layers') - 1,
                           input_shape=[1, 2]))
        model.add(Dropout(rate=hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(X_train_rnn, y_train, epochs=10, validation_data=(X_test_rnn, y_test))

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best RNN/LSTM Type: {best_hps.get('model_type')}")
print(f"Best Number of Layers: {best_hps.get('n_layers')}")
print(f"Best Number of Units: {best_hps.get('units')}")
print(f"Best Dropout Rate: {best_hps.get('dropout')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_rnn, y_train, epochs=50, validation_data=(X_test_rnn, y_test))

# Plotting training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('mn.png')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test_rnn, y_test)

print(f"Test Loss: {loss}")

print(f"Test Accuracy: {accuracy}")

# Predict the test data
y_pred = model.predict(X_test_rnn)
y_pred = (y_pred > 0.5).astype(int)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()].reshape(-1, 1, 2))  # Reshape input
Z = (Z > 0.5).astype(int)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='b', label='Class 0')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='r', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.legend()
plt.savefig('mn_db.png')  # Correct method to save a plot
plt.show()