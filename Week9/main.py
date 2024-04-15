import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model constructor function
def create_model(learning_rate=0.001, dropout_rate=0.2, units=128, activation='relu'):
    model = Sequential([
        Conv2D(32, (3, 3), activation=activation, padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=activation, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation=activation, padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(units, activation=activation),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model using KerasClassifier without directly passing 'units'
model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)

# Define the hyperparameters to tune
param_dist = {
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'model__units': [64, 128, 256],
    'model__activation': ['relu', 'tanh', 'sigmoid']
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)
random_search.fit(X_train, y_train.ravel())

# Results
best_params = random_search.best_params_
print(f"Best Hyperparameters: {best_params}")
# Build the model with the best hyperparameters
best_model = create_model(**best_params)

try:
    history = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=2)
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
except Exception as e:
    print(f"An error occurred: {e}")

# Evaluate the model on test data
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions
y_pred = best_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Calculate F1 score and recall
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
