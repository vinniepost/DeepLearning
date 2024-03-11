import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from sklearn.model_selection import train_test_split
from tensorflow import keras


from tensorflow.keras.callbacks import TensorBoard
log_dir = "./logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


def create_dataset(n_seq, seq_len):
    X = np.random.randint(0, 100, size=(n_seq, seq_len))
    y = X[:,-1]
    return X,y

X, y = create_dataset(1000, 100)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Implement a simple RNN

RNN_model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), # De none in inputlayer betekent dat er geen vaste groote is
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

RNN_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc}')

# Implement a simple LSTM

LSTM_model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20),
    keras.layers.Dense(1)
])

LSTM_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# LSTM_model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])
# test_loss, test_acc = LSTM_model.evaluate(X_test, y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc}')



# generate X -> y^2 datapoints with scikit-learn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

RNN_model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])
test_loss, test_acc = RNN_model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

LSTM_model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])
test_loss, test_acc = LSTM_model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# use matplotlib to show some of the sequences and their predictions for both the RNN and LSTM
plt.plot(RNN_model.predict(X_test[:10]), label='RNN')
plt.plot(LSTM_model.predict(X_test[:10]), label='LSTM')
plt.plot(y_test[:10], label='True')
plt.legend()
plt.show()
plt.savefig('RNN_LSTM.png')  # Save the plot as a .png file


