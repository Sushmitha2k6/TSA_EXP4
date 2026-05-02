# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02-05-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =========================
# Load Dataset
# =========================
df = pd.read_csv('/content/amazon_sales_dataset.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# 🔥 Combine duplicate dates
df = df.groupby('order_date')['total_revenue'].sum().to_frame()

# Sort index
df = df.sort_index()

# =========================
# Use time series data
# =========================
X = df['total_revenue']

# Plot original data
plt.figure(figsize=(12,6))
plt.plot(X)
plt.title('Amazon Revenue Time Series')
plt.show()

# =========================
# ACF & PACF
# =========================
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plot_acf(X, lags=min(30, len(X)//2), ax=plt.gca())
plt.title('ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=min(30, len(X)//2), ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()

# =========================
# ARMA(1,1)
# =========================
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

print("ARMA(1,1) parameters:")
print("phi1 =", phi1)
print("theta1 =", theta1)

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

arma1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.plot(arma1)
plt.title('Simulated ARMA(1,1)')
plt.show()

plot_acf(arma1)
plt.show()

plot_pacf(arma1)
plt.show()

# =========================
# ARMA(2,2)
# =========================
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

print("\nARMA(2,2) parameters:")
print("phi1 =", phi1, "phi2 =", phi2)
print("theta1 =", theta1, "theta2 =", theta2)

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

arma2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.plot(arma2)
plt.title('Simulated ARMA(2,2)')
plt.show()

plot_acf(arma2)
plt.show()

plot_pacf(arma2)
plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="569" height="489" alt="image" src="https://github.com/user-attachments/assets/7ea24a94-68af-4f9d-8813-abf26fd440d7" />


Partial Autocorrelation

<img width="598" height="451" alt="image" src="https://github.com/user-attachments/assets/78fa9aea-b333-4646-b902-f9df1ec8dee8" />


Autocorrelation

<img width="594" height="405" alt="image" src="https://github.com/user-attachments/assets/55b65bef-0734-43bb-9b4c-6e91d6f1d165" />


SIMULATED ARMA(2,2) PROCESS:


<img width="565" height="488" alt="image" src="https://github.com/user-attachments/assets/9f4258a4-d073-4260-a8c7-a8d7c8611360" />

Partial Autocorrelation


<img width="573" height="454" alt="image" src="https://github.com/user-attachments/assets/590514b9-f411-417a-be85-16032ce9a76b" />

Autocorrelation
<img width="578" height="436" alt="image" src="https://github.com/user-attachments/assets/2150b036-b844-49da-bd29-3233788a38d4" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
