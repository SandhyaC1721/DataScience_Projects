# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 2: Load dataset
data = pd.read_csv("temperature.csv", parse_dates=["Date"], index_col="Date")



# Step 3: Show first 5 rows
print(data.head())

# Step 4: Plot the data
data.plot()
plt.title("Daily Temperature Data")
plt.show()

# Step 5: Create ARIMA model
model = ARIMA(data, order=(5,1,0))

# Step 6: Train model
model_fit = model.fit()

# Step 7: Print model summary
print(model_fit.summary())

# Step 8: Forecast next 7 days
forecast = model_fit.forecast(steps=7)

# Step 9: Print forecast
print("Next 7 Days Prediction:")
print(forecast)