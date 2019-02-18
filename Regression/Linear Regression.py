import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import  numpy as np
# Load data
df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\Grade_Set_1.csv')
print(df)
# Simple scatter plot
# df.plot(kind='scatter', x='Hours_Studied', y='Test_Grade', title='Grade vs Hours Studied')
# plt.show()
# check the correlation between variables
print(df.corr())

lr = lm.LinearRegression()
x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade.values # dependent variable
lr.fit(x, y)
print("Intercept: ", lr.intercept_)
print("Coefficient: ", lr.coef_)
print("Using predict function: ", lr.predict(6))
# plotting fitted line
plt.scatter(x, y, color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title('Grade vs Hours Studied')
plt.ylabel('Test_Grade')
plt.xlabel('Hours_Studied')
plt.show()

# How good model is :-
#  3 Metrics : R-Squared , RMSE , MAE
#   R-Squared : R-squared value designates the total proportion of variance in the dependent
#     variable explained by the independent variable. It is a value between 0 and 1; the value
#     toward 1 indicates a better model fit.
#     R-Sqaured = (SumOfSqauresOf(yi^ - Ymean) / SumOfSquaresOf(yi - Ymean)
#     R-Sqaured = (Total Sum of Square Residuals (SSR)) / Sum of Square Total (SST))
#     if the R-Squared value is 0.97 then , that can be interpreted as - 97% variability in the dependant variable can be explained by the independant variable
#   RMSE : Square root of the mean of the  squared errors. RMSE indicates  how close the predicted values  are to the actual values. Hence low RMSE signifies that the
#     model performance is good
# 	  RMSE = SqrtOf(1/n * (SumOfSquaresOf(yi - Ymean)))
#   MAE :  mean or average of absolute value of the errors, that is, the predicted - actual
#     MAE = SqrtOf(1/n * (SumOf(| yi- Y^ |)))

# function to calculate r-squared, MAE, RMSE
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error
df['Test_Grade_Pred'] = lr.predict(x)
print("R Squared : ", r2_score(df.Test_Grade, y))
print("Mean Absolute Error: ", mean_absolute_error(df.Test_Grade, df.Test_Grade_Pred))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(df.Test_Grade,df.Test_Grade_Pred)))



