import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('./DATASET/android_traffic.csv', sep=';')
X = df.drop('type', axis=1).to_numpy()
y = np.array([1 if t == 'malware' else 0 for t in df['type']])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(1,5), activation='relu', input_shape=(1, X_train.shape[2], 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(1,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1,5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
