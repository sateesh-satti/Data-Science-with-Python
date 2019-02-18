# Polynomial regression : form of higher-order linear regression modeled between dependent and
#    independent variables as an nth degree polynomial

# Quadratic - Y = m1 X + m2 X^2 +c
# Cubic - Y = m1 X + m2 X^2 + m3 X^3 + c
# Nth - Y = m1 X + m2 X^2 + m3 X^3 + ... + mn X^n +c

# PLotting for ploynomial regression

import numpy as np
import matplotlib as plt
x = np.linspace(-3,3,1000) # 1000 sample number between -3 to 3
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
ax1.plot(x, x)
ax1.set_title('linear')
ax2.plot(x, x**2)
ax2.set_title('degree 2')
ax3.plot(x, x**3)
ax3.set_title('degree 3')
ax4.plot(x, x**4)
ax4.set_title('degree 4')
ax5.plot(x, x**5)
ax5.set_title('degree 5')
ax6.plot(x, x**6)
ax6.set_title('degree 6')
plt.tight_layout()# tidy layout
plt.show()


# Polynomial Regression Example
import pandas as pd
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\Grade_Set_2.csv')
print(df)
df.plot(kind='scatter', x='Hours_Studied', y='Test_Grade', title='Grade vs Hours Studied')
plt.show()
print("Correlation Matrix: ")
df.corr()

lr = lm.LinearRegression()
x= df.Hours_Studied[:, np.newaxis]  # independent variable
y= df.Test_Grade    # dependent variable
# Train the model using the training sets
lr.fit(x, y)
# plotting fitted line
plt.scatter(x, y, color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title('Grade vs Hours Studied')
plt.ylabel('Test_Grade')
plt.xlabel('Hours_Studied')
plt.show()
print("R Squared: ", r2_score(y, lr.predict(x)))


# R-Squared for different Polynomial degrees

lr = lm.LinearRegression()
x= df.Hours_Studied # independent variable
y= df.Test_Grade # dependent variable
# NumPy's vander function will return powers of the input vector
for deg in [1, 2, 3, 4, 5]:
	lr.fit(np.vander(x,deg+1),y)
	y_lr = lr.predict(np.vander(x, deg + 1))
	plt.plot(x, y_lr, label='degree ' + str(deg));
	plt.legend(loc=2);
	print(r2_score(y, y_lr))
plt.plot(x,y,'ok')
plt.show()


# sklearn provides a function to generate a new feature matrix  consisting  of all polynomial combinations of the features  with the
# features with the degree less than or equal to specified degree

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade # dependent variable
degree = 4 # it will increase upto 3
model = make_pipeline(PolynomialFeatures(degree), lr)
model.fit(x, y)
plt.scatter(x, y, color='black')
plt.plot(x, model.predict(x), color='green')
print("R Squared using built-in function: ", r2_score(y, model.predict(x)))
plt.show()


