# Predicted probability is a number between 0 and 1. Traditionally >.5 is the cutoff point
# used for converting predicted probability to 1 (positive), otherwise it is 0 (negative). This
# logic works well when your training dataset has equal examples of positive and negative
# cases; however this is not the case in real-world scenarios.

# The solution is to find the optimal cutoff point, that is, the point where the true
# positive rate is high and the false positive rate is low. Anything above this threshold can
# be labeled as 1 or else it is 0.

import pandas as pd
import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# read the data in
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
# target variable % distribution
print(df['class'].value_counts(normalize=True))
X = df.ix[:,:8] # independent variables
y = df['class'] # dependent variables

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
# instantiate a logistic regression model, and fit
model = LogisticRegression()
model = model.fit(X_train, y_train)
# predict class labels for the train set. The predict fuction converts probability values > .5 to 1 else 0
y_pred = model.predict(X_train)

# generate class probabilities
# Notice that 2 elements will be returned in probs array,
# 1st element is probability for negative class,
# 2nd element gives probability for positive class
probs = model.predict_proba(X_train)
y_pred_prob = probs[:, 1]
print("Accuracy: ", metrics.accuracy_score(y_train, y_pred))
# The optimal cutoff would be where the true positive rate (tpr) is high and the
# false positive rate (fpr) is low, and tpr - (1-fpr) is zero or near to zero.

# extract false positive, true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame(
		{'fpr' : pd.Series(fpr, index=i),
		 'tpr' : pd.Series(tpr,index = i),
		 '1-fpr' : pd.Series(1-fpr, index = i),
		 'tf' : pd.Series(tpr - (1-fpr), index = i),
		 'thresholds' : pd.Series(thresholds, index = i)}
	)
roc.ix[(roc.tf-0).abs().argsort()[:1]]
# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'], label='tpr')
plt.plot(roc['1-fpr'], color = 'red', label='1-fpr')
plt.legend(loc='best')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

def Find_Optimal_Cutoff(target, predicted):
	fpr , tpr , threshold = metrics.roc_curve(target , predicted)
	i = np.arange(len(tpr))
	roc = pd.DataFrame(
			{
				'tf':pd.Series(tpr - (1 - fpr) , index = i) ,
				'threshold':		pd.Series(threshold , index = i)
			})
	roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
	return list(roc_t['threshold'])

threshold = Find_Optimal_Cutoff(y_train, probs[:, 1])
print("Optimal Probability Threshold: ", threshold)

# Applying the threshold to the prediction probability
y_pred_optimal = np.where(y_pred_prob >= threshold, 1, 0)

print("\nNormal - Accuracy: ", metrics.accuracy_score(y_train, y_pred))
print("Optimal Cutoff - Accuracy: ", metrics.accuracy_score(y_train, (y_pred_optimal)))
print("\nNormal - Confusion Matrix: \n", metrics.confusion_matrix(y_train, y_pred))
print("Optimal - Cutoff Confusion Matrix: \n", metrics.confusion_matrix(y_train, y_pred_optimal))
