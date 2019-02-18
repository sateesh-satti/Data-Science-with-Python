# ARIMA - Auto Regressive Integrated Moving Average

# data points that are collected sequentially at a regular interval with
# association over a time period is termed time-series data

# A time-series data having the
# mean and variance as a constant is called a stationary time series

# Time series tend to have a linear relationship between lagged variables and this is
# called an autocorrelation. Hence a time series historic data can be modeled to forecast
# the future data points without involvement of any other independent variables; these
# types of models are generally known as time-series forecasting

# Components of the TIME SERIES :
# time series can be made up of three key components :-

# Trend – A long-term increase or decrease are termed trends
# Seasonality - An effect of seasonal factors for a fixed or known
#   period. For example, retail stores sales will be high during
#   weekends and festival seasons.
# Cycle – These are the longer ups and downs that are not of fixed or
#   known periods caused by external factors.

# Auto Regressive (AR) :- As the Name indicates, It is the regression of the variable itself,i.e,
#   the linear combination of past values of the variables are used to forecast the future value

# Moving Average (MR) :- Instead of past values, A past forecast errors are used to build the model

# Autoregressive (AR), moving average (MA) model with integration (opposite of
#   differencing) is called the ARIMA model

# Page no. 204 of Mastering Machine learning Book

# The predictors on the right side of the equation are the lagged values, errors, and
# it is also known as ARIMA (p, d, q) model. These are the key parameters of ARIMA and
# picking the right value for p, d, q will yield better model results.
# p = order of the autoregressive part. That is the number of unknown terms that
#   multiply your signal at past times (so many past times as your value p).
# d = degree of first differencing involved. Number of times you have to difference your
#   time series to have a stationary one.
# q = order of the moving average part. That is the number of unknown terms that
#   multiply your forecast errors at past times (so many past times as your value q).

# Steps for Running ARIMA Model
# • Plot the chart to ensure trend, cycle, or seasonality exists in the dataset.
# • Stationarize series: To stationarize series we need to remove trend
#    (varying mean) and seasonality (variance) components from the
#    series. Moving average and differencing technique can be used to
#    stabilize trend, whereas log transform will stabilize the seasonality
#    variance. Further, the Dickey Fuller test can be used to assess the
#    stationarity of series, that is, null hypothesis for a Dickey Fuller
#    test is that the data are stationary, so test result with p value > 0.05
#    means data is non-stationary.
# • Find optimal parameter: Once the series is stationarized you
#    can look at the Autocorrelation function (ACF) and Partial
#    autocorrelation function (PACF) graphical plot to pick the
#    number of AR or MA terms needed to remove autocorrelation.
#    ACF is a bar chart between correlation coefficients and lags;
#    similarly PACF is the bar chart between partial correlation
#    (correlation between variable and lag of itself not explained by
#    correlation at all lower-order lags) coefficient and lags.
# • Build Model and Evaluate: Since time series is a continuous
#    number Mean Absolute Error and Root Mean Squared Error can
#    be used to evaluate the deviation between actual and predicted
#    values in train dataset. Other useful matrices would be Akaike
#    Information Criterion (AIC) and Bayesian Information Criterion
#    (BIC); these are part of information theory to estimate the quality
#    of individual models given a collection of models, and they favor a
#    model with smaller residual errors.
#    AIC = -2log(L) + 2(p+q+k+1) where L is the maximum
#    likelihood function of fitted model and p, q, k are the number
#    of parameters in the model
#    BIC = AIC+(log(T)−2)(p+q+k+1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\\Supervised Learning – Classification\\Data\\TS.csv',engine='python')
ts = pd.Series(list(df['Sales']), index=pd.to_datetime(df['Month'],format='%Y-%m'))

from statsmodels.tsa.seasonal import seasonal_decompose
ts_log = np.log(ts)
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

ts_log.dropna(inplace=True)
s_test = adfuller(ts_log, autolag='AIC')
print("Log transform stationary check p value: ", s_test[1])

#Take first difference:
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
plt.title('Trend removed plot with first order difference')
plt.plot(ts_log_diff)
plt.ylabel('First order log diff')
plt.show()
s_test = adfuller(ts_log_diff, autolag='AIC')
print("First order difference stationary check p value: ", s_test[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,3))
# ACF chart
fig = sm.graphics.tsa.plot_acf(ts_log_diff.values.squeeze(), lags=20,ax=ax1)


# draw 95% confidence interval line
ax1.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax1.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax1.set_xlabel('Lags')

# PACF chart
fig = sm.graphics.tsa.plot_pacf(ts_log_diff, lags=20, ax=ax2)
# draw 95% confidence interval line
ax2.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax2.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax2.set_xlabel('Lags')
plt.show()

# PACF plot has a significant spike only at lag 1, meaning that all the higher-order
#    autocorrelations are effectively explained by the lag-1 and lag-2 autocorrelation.
# Ideal lag values are p = 2 and q = 2, that is, the lag value where the ACF/PACF chart crosses the
#    upper confidence interval for the first time.

# Building the Model :-

model = sm.tsa.ARIMA(ts_log, order=(2,0,2))
results_ARIMA = model.fit(disp=-1)
ts_predict = results_ARIMA.predict()
plt.title('ARIMA Prediction')
plt.plot(ts_log, label='Actual')
plt.plot(ts_predict, 'r--', label='Predicted')
plt.xlabel('Year-Month')
plt.ylabel('Sales')
plt.legend(loc='best')
plt.show()
# Evaluate model
print( "AIC: ", results_ARIMA.aic)
print( "BIC: ", results_ARIMA.bic)

print("Mean Absolute Error: ", mean_absolute_error(ts_log.values, ts_predict.values))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(ts_log.values,ts_predict.values)))
print("Durbin-Watson statistic :", sm.stats.durbin_watson(results_ARIMA.resid.values))


model = sm.tsa.ARIMA(ts_log, order=(3,0,2))
results_ARIMA = model.fit(disp=-1)
# ts_predict = results_ARIMA.predict('1965-01-01', '1972-05-01',dynamic=True)
ts_predict = results_ARIMA.predict()
plt.title('ARIMA Prediction')
plt.plot(ts_log, label='Actual')
plt.plot(ts_predict, 'r--', label='Predicted')
plt.xlabel('Year-Month')
plt.ylabel('Sales')
plt.legend(loc='best')
plt.show()
print( "AIC: ", results_ARIMA.aic)
print( "BIC: ", results_ARIMA.bic)

print("Mean Absolute Error: ", mean_absolute_error(ts_log.values, ts_predict.values))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(ts_log.values,ts_predict.values)))
print("Durbin-Watson statistic :", sm.stats.durbin_watson(results_ARIMA.resid.values))

model = sm.tsa.ARIMA(ts_log, order=(3,1,2))
results_ARIMA = model.fit(disp=-1)
ts_predict = results_ARIMA.predict()
# Correctcion for difference
predictions_ARIMA_diff = pd.Series(ts_predict, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

plt.title('ARIMA Prediction - order(3,1,2)')
plt.plot(ts_log, label='Actual')
plt.plot(predictions_ARIMA_log, 'r--', label='Predicted')
plt.xlabel('Year-Month')
plt.ylabel('Sales')
plt.legend(loc='best')
plt.show()

print( "AIC: ", results_ARIMA.aic)
print( "BIC: ", results_ARIMA.bic)

print("Mean Absolute Error: ", mean_absolute_error(ts_log_diff.values, ts_predict.values))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(ts_log_diff.values, ts_predict.values)))
# check autocorrelation
print("Durbin-Watson statistic :", sm.stats.durbin_watson(results_ARIMA.resid.values))


# Note: AIC/BIC can be positive or negative; however we should look at the absolute value of it for evaluation

# Predicting the Future Values

# final model
model = sm.tsa.ARIMA(ts_log, order=(3,0,2))
results_ARIMA = model.fit(disp=-1)
# predict future values
ts_predict = results_ARIMA.predict('1971-06-01', '1972-05-01')
plt.title('ARIMA Future Value Prediction - order(3,0,2)')
plt.plot(ts_log, label='Actual')
plt.plot(ts_predict, 'r--', label='Predicted')
plt.xlabel('Year-Month')
plt.ylabel('Sales')
plt.legend(loc='best')
plt.show()



