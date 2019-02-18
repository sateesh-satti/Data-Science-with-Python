#   concept of having multiple independent variables is called multivariate regression

# y =  m1X1 + m2X2 + m3X3 +  mnXn
# each independent variable is represented by x’s, and m’s are the corresponding coefficients.

# Multicollinearity and Variation Inflation Factor (VIF)

# The dependent variable should have a strong relationship with independent variables.
# However, any independent variables should not have strong correlations among other independent variables

# Multicollinearity is an incident where one or more of the independent variables are strongly correlated with each other. In such incidents, we
# should use only one among correlated independent variables.

# VIF is an indicator of the existence of multicollinearity, and ‘statsmodel’ provides a function to calculate the VIF for each independent variable
# and a value of greater than
# 10 is the rule of thumb for possible existence of high multicollinearity. The standard guideline for VIF value is as follows,
# VIF = 1 means no correlation exists,
# VIF > 1, but < 5 means moderate correlation exists

import  pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\'
                 'Housing_Modified.csv')
# Convert binary fields to numeric boolean fields
lb = preprocessing.LabelBinarizer()

df.driveway = lb.fit_transform(df.driveway)
df.recroom = lb.fit_transform(df.recroom)
df.fullbase = lb.fit_transform(df.fullbase)
df.gashw = lb.fit_transform(df.gashw)
df.airco = lb.fit_transform(df.airco)
df.prefarea = lb.fit_transform(df.prefarea)

# Create dummy variables for stories
df_stories = pd.get_dummies(df['stories'], prefix='stories', drop_first=True)

# Join the dummy variables to the main dataframe
df = pd.concat([df, df_stories], axis=1)
del df['stories']

# lets plot correlation matrix using statmodels graphics packages's plot_corr
# create correlation matrix
corr = df.corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()

# create a Python list of feature names
independent_variables = ['lotsize', 'bedrooms', 'bathrms','driveway','recroom', 'fullbase','gashw','airco','garagepl', 'prefarea', 'stories_one',
                         'stories_two','stories_three']

# use the list to select a subset from original DataFrame
X = df[independent_variables]
y = df['price']

thresh = 10

for i in np.arange(0,len(independent_variables)):
	vif = [variance_inflation_factor(X[independent_variables].values, ix) for ix in range(X[independent_variables].shape[1])]
	maxloc = vif.index(max(vif))
	if max(vif) > thresh:
		print("vif :", vif)
		print('dropping \'' + X[independent_variables].columns[maxloc] + '\'at index: ' + str(maxloc))
		del independent_variables[maxloc]
	else:
		break
# Removed bedrooms because it has high VIF > 10

# create a Python list of feature names (New Features after removing the high VIF variable
independent_variables = ['lotsize', 'bathrms','driveway', 'recroom', 'fullbase','gashw','airco','garagepl', 'prefarea', 'stories_one','stories_two',
                         'stories_three']
# use the list to select a subset from original DataFrame
X = df[independent_variables]
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80,random_state=1)
# ------------Using OLS of Statmodels------------
# create a fitted model & print the summary
lm = sm.OLS(y_train, X_train).fit()
print(lm.summary())

# make predictions on the testing set
y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

print("Train MAE: ", mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Interpretation of OLS regeression Results
# Adjusted R-Square :- Simple R-squared value will keep increase with addition of independent variable. To fix this issue adjusted R-squared is
#    considered for multivariate regression to understand the explanatory power of the independent variables
#    - With inclusion of more variables R-squared always tend to increase
#    - Adjusted R-squared will drop if the variable added does not explain the variable in the dependent variable

# Coeffecient :- These are the individual coefficients for respective independent variables. It can be either a positive or negative number,
#    which indicates that an increase in every unit of that independent variable will have a positive or negative impact on the dependent variable value.

# Standard error: This is the average distance of the respective independent observed values from the regression line.
#    The smaller values show that the model fitting is good.

# Durbin-Watson: It’s one of the common statistics used to determine the existence of multicollinearity, which means two or more independent variables
#    used in the multivariate regression model are highly correlated. The Durbin-Watson statistics are always between the number 0 and 4.
#    A value around 2 is ideal (range of 1.5 to 2.5 is relatively normal), and it means that there is no autocorrelation between the variables
#    used in the model.

# Confidence interval: This is the coefficient to calculate 95% confidence interval for the independent variable’s slope.

# p-value ≤ 0.05 signifies strong evidence against the null hypothesis, so you reject the null hypothesis. A p-value > 0.05 signifies
#    weak evidence against the null hypothesis, so you fail to reject the null hypothesis.


# Regression Diagnosis :-

# 1) outliers :- Data points that are far away from the fitted regression line are called outliers, and these can impact the accuracy of the model.
# Plotting normalized residual vs. leverage will give us a good understanding of the outliers points. Residual is the difference
# between actual vs. predicted, and leverage is a measure of how far away the independent variable values of an observation are from
# those of the other observations.

from statsmodels.graphics.regressionplots import plot_leverage_resid2
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(lm, ax = ax)
plt.show()

# Running a Bonferroni outlier test will give us p-values for each observation, and
#   those observations with p value < 0.05 are the outliers affecting the accuracy.

# Find outliers #
# Bonferroni outlier test
test = lm.outlier_test()
print('Bad data points (bonf(p) < 0.05):')
print(test[test.iloc[:,2] < 0.05])

# 2) Homoscedasticity and Normality :- The error variance should be constant, which is known has homoscedasticity and the error should be normally distributed

# plot to check homoscedasticity
plt.plot(lm.resid,'o')
plt.title('Residual Plot')
plt.ylabel('Residual')
plt.xlabel('Observation Numbers')
plt.show()
plt.hist(lm.resid, density=True)
plt.show()

# Linearity – the relationships between the predictors and the outcome variables should be linear. If the relationship is not linear then appropriate
# transformation (such as log, square root, and higher-order polynomials etc) should be applied to the dependent/independent variable to fix the issue.

# linearity plots
fig = plt.figure(figsize=(10,15))
fig = sm.graphics.plot_partregress_grid(lm, fig=fig)
plt.show()

# 3) UnderFitting and OverFitting :-
#   Underfitting - Under-fitting occurs when the model does not fit the data well and is unable to capture the underlying trend in it.
#       In this case we can notice a low accuracy in training and test dataset.
#   OverFitting - over-fitting occurs when the model fits the data  too well, capturing all the noises.
#       In this case we can notice a high accuracy in the training dataset, whereas the same model will result in a low accuracy on the test dataset

# 4) Regulariation :- With an increase in number of variables, and increase in model complexity, the probability
#        of over-fitting also increases. Regularization is a technique to avoid the over-fitting problem.
#        Statsmodel and the scikit-learn provides Ridge and LASSO (Least Absolute Shrinkage and Selection Operator) regression to handle the over-fitting issue.
#         With an increase in model complexity, the size of coefficients increase exponentially, so the ridge and LASSO
#        regression apply penalty to the magnitude of the coefficient to handle the issue.
# LASSO: This provides a sparse solution, also known as L1 regularization. It guides parameter value to be zero, that is, the coefficients of the variables
#       that add minor value to the model will be zero, and it adds a penalty equivalent to absolute value of the magnitude of coefficients.
# Ridge Regression: Also known as Tikhonov (L2) regularization, it guides parameters to be close to zero, but not zero. You can use this when you have many
#        variables that add minor value to the model accuracy individually; however it improves overall the model accuracy
#       and cannot be excluded from the model. Ridge regression will apply a penalty to reduce the magnitude of the coefficient of all variables
#       that add minor value to the model accuracy, and which adds penalty equivalent to square of the magnitude of coefficients. Alpha is the
#        regularization strength and must be a positive float.

from sklearn import linear_model
# Load data
df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\Grade_Set_2.csv')
df.columns = ['x','y']
for i in range(2,50):# power of 1 is already there
	colname = 'x_%d'%i  # new var will be x_power
	df[colname] = df['x']**i
independent_variables = list(df.columns)
independent_variables.remove('y')
X= df[independent_variables] # independent variable
y= df.y # dependent variable
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80,test_size = .20,random_state=1)

# Ridge regression
lr = linear_model.Ridge(alpha=0.001)
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
print("------ Ridge Regression ------")
print("Train MAE: ", mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Ridge Coef: ", lr.coef_)

# LASSO regression
lr = linear_model.Lasso(alpha=0.001)
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
print("----- LASSO Regression -----")
print("Train MAE: ", mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("LASSO Coef: ", lr.coef_)





