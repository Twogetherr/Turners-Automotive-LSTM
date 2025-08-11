import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


ITERATIONS = 60 # The amount of times model will train on data.
SEQUENCE_LENGTH = 60 # Days in the past model trains from.
LSTM_MEMORY_LAYERS = 50 # Number of memory cell units.
BATCH_SIZE = 32 # Sample size model processes at once.
TRAINING_DATA_PERCENTAGE = 0.8 # Percentage of data used for training.


df = pd.read_csv("NZX_TRA.csv")

# Check NZX_TRA.csv for features that can be used.
feature_cols = ['open', 'high', 'low', 'close', 'Volume', 'EMA', 'RSI', 'MACD', 'ATR']

early_stop = EarlyStopping(monitor='val_loss', patience=5)
features = df[feature_cols]
target = df['close'] # Estimates the closing price

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target.values.reshape(-1,1))

def create_sequences(features, target, SEQUENCE_LENGTH):
    X = []
    y = []
    for i in range(SEQUENCE_LENGTH, len(features)):
        X.append(features[i-SEQUENCE_LENGTH:i])
        y.append(target[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target, SEQUENCE_LENGTH)
train_size = int(TRAINING_DATA_PERCENTAGE * len(X)) 

print("Input shape for LSTM:", X.shape)  # (samples, timesteps, features)
print("Target shape:", y.shape)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

model = models.Sequential([
    layers.LSTM(LSTM_MEMORY_LAYERS, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(LSTM_MEMORY_LAYERS),
    layers.Dense(1)  # Predict the next day’s close price
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=ITERATIONS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    #callbacks=[early_stop],
    shuffle=False
)

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

y_pred = model.predict(X_test)

y_test_real = target_scaler.inverse_transform(y_test)
y_pred_real = target_scaler.inverse_transform(y_pred)

# Saves your model.
#joblib.dump(feature_scaler, 'feature_scaler.pkl')
#joblib.dump(target_scaler, 'target_scaler.pkl')
#joblib.dump(model, 'my_model.pkl')


# Plot against data
plt.figure(figsize=(14,7))
plt.plot(y_test_real, label='Actual Price')
plt.plot(y_pred_real, label='Predicted Price')
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Time Steps (Test Samples)')
plt.ylabel('Price')
plt.legend()
plt.show()

