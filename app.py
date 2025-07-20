import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('📈 Stock Trend Prediction')

# User input
user_input = st.text_input("Enter Stock Ticker (e.g. AAPL, GOOG, INFY)", 'AAPL')
df = yf.download(user_input, start='2020-01-01', end='2024-12-31', auto_adjust=True)

# Describing Data
st.subheader('Data from 2020-2024')
st.write(df.describe())

# Visualization
st.subheader('📊 Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('📊 Closing Price with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label="100 MA")
plt.plot(df.Close, label="Closing Price")
plt.legend()
st.pyplot(fig)

st.subheader('📊 Closing Price with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label="100 MA")
plt.plot(ma200, 'g', label="200 MA")
plt.plot(df.Close, 'b', label="Closing Price")
plt.legend()
st.pyplot(fig)

# Splitting Data
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('stock_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

# Predict
y_predicted = model.predict(x_test)

# Fix shape if model returns sequences (like (n, 100, 1))
if len(y_predicted.shape) == 3:
    y_predicted = y_predicted[:, -1, 0]  # take the last time step
else:
    y_predicted = y_predicted.reshape(-1)

y_test = y_test.reshape(-1)

# Inverse transform
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting Predictions
st.subheader('📈 Predicted vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Original Price', color='blue')
plt.plot(y_predicted, label='Predicted Price', color='red')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.title('Original vs Predicted Stock Price')
plt.legend()
plt.grid(True)
st.pyplot(fig2)
