import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 60 # Should be same as testing.

model = joblib.load('my_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

new_data = pd.read_csv('new_dataset.csv')

# Feature columns must be same as what was used in training.
feature_columns = ['open', 'high', 'low', 'close', 'Volume', 'EMA', 'RSI', 'MACD', 'ATR'] 
new_features = new_data[feature_columns].values

scaled_features = feature_scaler.transform(new_features)

sequence_length = SEQUENCE_LENGTH
X_new = []

for i in range(sequence_length, len(scaled_features)):
    X_new.append(scaled_features[i-sequence_length:i])

X_new = np.array(X_new)

scaled_predictions = model.predict(X_new)

predictions = target_scaler.inverse_transform(scaled_predictions)

print(predictions)

# Makes a new csv
pd.DataFrame(predictions, columns=['Predicted_Close']).to_csv('predictions.csv', index=False)

# Plots the estimation.
plt.figure(figsize=(14,7))
plt.plot(predictions, label='Predicted Price')
plt.title('Predicted Close Price')
plt.xlabel('Working Day After Data')
plt.ylabel('Price')
plt.legend()
plt.show()