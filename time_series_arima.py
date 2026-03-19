import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample time series data
data = {
    "Date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
    "Value": [10, 12, 13, 15, 18, 17, 19, 21, 23, 22,
              24, 26, 28, 27, 29, 30, 32, 31, 33, 35,
              36, 38, 37, 39, 40, 42, 43, 45, 44, 46]
}

df = pd.DataFrame(data)
df.set_index("Date", inplace=True)

# Train ARIMA model
model = ARIMA(df["Value"], order=(2,1,2))
model_fit = model.fit()

# Forecast next 5 days
forecast = model_fit.forecast(steps=5)

# Plot
plt.figure()
plt.plot(df.index, df["Value"], label="Original Data")
plt.plot(pd.date_range(df.index[-1], periods=6, freq="D")[1:], forecast, label="Forecast")
plt.legend()
plt.title("Time Series Forecasting using ARIMA")
plt.show()
