# The fundamental idea here is that the hypothesis will use the linear approximation, then map it with a logistic function for binary prediction.

#  Logistic regression can be explained better in odds ratio. The odds of an event occurring are defined as the
#  probability of an event occurring divided by the probability of that event not occurring.

# odds ratio of pass vs fail = probability (y =1)/1- probability (y =1)

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# manually add intercept

df = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\'
                 'Grade_Set_1_Classification.csv')
df['intercept'] = 1
independent_variables = ['Hours_Studied', 'intercept']
x = df[independent_variables] # independent variable
y = df['Result']

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(x, y)

# check the accuracy on the training set
model.score(x, y)
print(model.predict(x))
print(model.predict_proba(x)[:,0])

# plotting fitted line
plt.scatter(df.Hours_Studied, y, color='black')
plt.yticks([0.0, 0.5, 1.0])
plt.plot(df.Hours_Studied, model.predict_proba(x)[:,1], color='blue',linewidth=3)
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()

# Evaluating the classification model performance

# Confusion matrix is the table that is used for describing the performance of a classification model

#                       Predicted
#                      ------------
#                    | False |True |
#                    --------------
#           False   | TN   |   FP |
# Actual           ---------------
#           True  |  FN   |   TP |
#                 ---------------


# True Negatives (TN): Actual FALSE, which was predicted as FALSE
# False Positives (FP): Actual FALSE, which was predicted as TRUE (Type I error)
# False Negatives (FN): Actual TRUE, which was predicted as FALSE (Type II error)
# True Positives (TP): Actual TRUE, which was predicted as TRUE

# Ideally a good model should have high TN and TP and less of Type I & II errors

# Classification performance matrices

# Accuracy - what % of predictions were correct? - (TP+TN)/(TP+TN+FP+FN)
# Misclassification Rate - what % of prediction is wrong? - (FP+FN)/(TP+TN+FP+FN)
# True Positive Rate OR Sensitivity OR Recall (completeness) - what % of positive cases did model catch? - TP/(FN+TP)
# False Positive Rate - what % of 'No' were predicted as 'Yes'? - FP/(FP+TN)
# Specificity - what % of 'No' were predicted as 'No'?  - TN/(TN+FP)
# Precision (exactness) - what % of positive predictions were correct? - TP/(TP+FP)
# F1 score - Weighted average of precision and recall - 2*((precision * recall) / (precision + recall))


from sklearn import metrics
# generate evaluation metrics
print("Accuracy :", metrics.accuracy_score(y, model.predict(x)))
print("AUC :", metrics.roc_auc_score(y, model.predict_proba(x)[:,1]))
print("Confusion matrix :\n",metrics.confusion_matrix(y, model.predict(x)))
print("classification report :\n", metrics.classification_report(y, model.predict(x)))


# ROC :- A ROC curve is one more important metric, and it’s a most commonly used way to visualize the performance of a binary classifier
# AUC :- AUC is believed to be one of the best ways to summarize performance in a single number.
# AUC indicates that the probability of a randomly selected positive example will be scored higher by the classifier than a randomly selected negative example.
# If you have multiple models with nearly the same
# accuracy, you can pick the one that gives a higher AUC.

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y, model.predict_proba(x)[:,1])
# Calculate the AUC
roc_auc = metrics.auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Fitting Line :- The inverse of regularization is one of the key aspects of fitting a logistic regression line. It defines the complexity of the fitted line.
# Changing the C value for the Logistic regression varies the fitting line (Default =1)

model = LogisticRegression()
model = model.fit(x, y)
#check the accuracy on the training set
print("C = 1 (default), Accuracy :", metrics.accuracy_score(y, model.predict(x)))
#instantiate a logistic regression model with c = 10, and fit with X and y
model1 = LogisticRegression(C=10)
model1 = model1.fit(x, y)
#check the accuracy on the training set
print("C = 10, Accuracy :", metrics.accuracy_score(y, model1.predict(x)))
#instantiate a logistic regression model with c = 100, and fit with X and y
model2 = LogisticRegression(C=100)
model2 = model2.fit(x, y)
#check the accuracy on the training set
print("C = 100, Accuracy :", metrics.accuracy_score(y, model2.predict(x)))
#instantiate a logistic regression model with c = 1000, and fit with X and y
model3 = LogisticRegression(C=1000)
model3 = model3.fit(x, y)
#check the accuracy on the training set
print("C = 1000, Accuracy :", metrics.accuracy_score(y, model3.predict(x)))
#plotting fitted line
plt.scatter(df.Hours_Studied, y, color='black', label='Result')
plt.yticks([0.0, 0.5, 1.0])
plt.plot(df.Hours_Studied, model.predict_proba(x)[:,1], color='gray',linewidth=2, label='C=1.0')
plt.plot(df.Hours_Studied, model1.predict_proba(x)[:,1], color='blue',linewidth=2,label='C=10')
plt.plot(df.Hours_Studied, model2.predict_proba(x)[:,1], color='green',linewidth=2,label='C=100')
plt.plot(df.Hours_Studied, model3.predict_proba(x)[:,1], color='red',linewidth=2,label='C=1000')
plt.legend(loc='lower right') # legend location
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()

# Regularization

# With an increase in the number of variables, the probability of over-fitting also increases.
# LASSO (L1) and Ridge (L2) can be applied for logistic regression as well to avoid overfitting.

import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\3 Machine Learning - Part 1\Regression\\Data\\LR_NonLinear.csv')
pos = data['class'] == 1
neg = data['class'] == 0
x1 = data['x1']
x2 = data['x2']
def draw_plot():
	plt.figure(figsize=(6, 6))
	plt.scatter(np.extract(pos, x1),np.extract(pos, x2),c='b', marker='s', label='pos')
	plt.scatter(np.extract(neg, x1),np.extract(neg, x2),c='r', marker='o', label='neg')
	plt.xlabel('x1');
	plt.ylabel('x2');
	plt.axes().set_aspect('equal', 'datalim')
	plt.legend();

# create hihger order polynomial for independent variables
order_no = 6

# map the variable 1 & 2 to its higher order polynomial
def map_features(variable_1, variable_2, order=order_no):
	assert order >= 1
	def iter():
		for i in range(1, order + 1):
			for j in range(i + 1):
				yield np.power(variable_1, i - j) * np.power(variable_2, j)
	return np.vstack(iter())

out = map_features(data['x1'], data['x2'], order=order_no)
X = out.transpose()
y = data['class']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

# function to draw classifier line
def draw_boundary(classifier):
	dim = np.linspace(-0.8, 1.1, 100)
	dx, dy = np.meshgrid(dim, dim)
	v = map_features(dx.flatten(), dy.flatten(), order=order_no)
	z = (np.dot(classifier.coef_, v) + classifier.intercept_).reshape(100, 100)
	plt.contour(dx, dy, z, levels=[0], colors=['r'])

# fit with c = 0.01
clf = LogisticRegression(C=0.01).fit(X_train, y_train)
print('Train Accuracy for C=0.01: ', clf.score(X_train, y_train))
print('Test Accuracy for C=0.01: ', clf.score(X_test, y_test))
draw_plot()
plt.title('Fitting with C=0.01')
draw_boundary(clf)
plt.legend();
plt.show()

# fit with c = 1
clf = LogisticRegression(C=1).fit(X_train, y_train)
print('Train Accuracy for C=1: ', clf.score(X_train, y_train))
print('Test Accuracy for C=1: ', clf.score(X_test, y_test))
draw_plot()
plt.title('Fitting with C=1')
draw_boundary(clf)
plt.legend();
plt.show()

# fit with c = 10000
clf = LogisticRegression(C=10000).fit(X_train, y_train)
print('Train Accuracy for C=10000: ', clf.score(X_train, y_train))
print('Test Accuracy for C=10000: ', clf.score(X_test, y_test))
draw_plot()
plt.title('Fitting with C=10000')
draw_boundary(clf)
plt.legend();
plt.show()


















