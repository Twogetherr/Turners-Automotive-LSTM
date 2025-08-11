import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("NZX_TRA.csv")
feature_cols = ['open', 'high', 'low', 'close', 'Volume', 'EMA', 'RSI', 'MACD', 'ATR']

features = df[feature_cols]
target = df['close']

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target.values.reshape(-1,1))

seq_length = 60
#early_stop = EarlyStopping(monitor='val_loss', patience=5)

def create_sequences(features, target, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(features)):
        X.append(features[i-seq_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target, seq_length)
train_size = int(0.8 * len(X))

print("Input shape for LSTM:", X.shape)  # (samples, timesteps, features)
print("Target shape:", y.shape)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(50),
    layers.Dense(1)  # Predict the next day’s close price
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.1,
    #callbacks=[early_stop],
    shuffle=False  # Important for time series data
)

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")


y_pred = model.predict(X_test)

y_test_real = target_scaler.inverse_transform(y_test)
y_pred_real = target_scaler.inverse_transform(y_pred)

joblib.dump(feature_scaler, 'feature_scaler.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
joblib.dump(model, 'my_model.pkl')

plt.figure(figsize=(14,7))
plt.plot(y_test_real, label='Actual Price')
plt.plot(y_pred_real, label='Predicted Price')
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Time Steps (Test Samples)')
plt.ylabel('Price')
plt.legend()
plt.show()

