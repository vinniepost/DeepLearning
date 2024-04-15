import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

# Generate moons dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the input data for RNN and LSTM
X_train_rnn = X_train.reshape(-1, 1, 2)
X_test_rnn = X_test.reshape(-1, 1, 2)

# Create RNN and LSTM models
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[1, 2]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[1, 2]),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the models
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the models
rnn_model.fit(X_train_rnn, y_train, epochs=10, validation_data=(X_test_rnn, y_test), callbacks=[tensorboard_callback])
lstm_model.fit(X_train_rnn, y_train, epochs=10, validation_data=(X_test_rnn, y_test), callbacks=[tensorboard_callback])

# Evaluate the models
rnn_model.evaluate(X_test_rnn, y_test)
lstm_model.evaluate(X_test_rnn, y_test)

# Make predictions
rnn_predictions = rnn_model.predict(X_test_rnn)
lstm_predictions = lstm_model.predict(X_test_rnn)

# Plot the predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=rnn_predictions[:, 0], cmap='coolwarm')
plt.title('RNN Predictions')
plt.colorbar(label='Probability')
# Plot decision boundary
x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = rnn_model.predict(np.c_[xx.ravel(), yy.ravel()].reshape(-1, 1, 2))  # Reshape for prediction
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=lstm_predictions[:, 0], cmap='coolwarm')
plt.title('LSTM Predictions')
plt.colorbar(label='Probability')
# Plot decision boundary
Z = lstm_model.predict(np.c_[xx.ravel(), yy.ravel()].reshape(-1, 1, 2))  # Reshape for prediction
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.savefig('rnn_lstm_predictions.png')
plt.show()