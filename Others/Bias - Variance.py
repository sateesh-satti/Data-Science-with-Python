# Bias :-
# If model accuracy is low on a training dataset as well as test dataset the model is said to
#   be under-fitting or that the model has high bias. This means the model is not fitting the
#   training dataset points well in regression or the decision boundary is not separating the
#   classes well in classification;
#    Two key reasons for bias are
#       1) not including the right features, and
#       2) not picking the correct order of polynomial degrees for model fitting.

# To solve an under-fitting issue or to reduced bias :-
#       try including more meaningful features and
#       try to increase the model complexity by trying higher-order polynomial fittings.

# Variance :-
# If a model is giving high accuracy on a training dataset, however on a test dataset the
# accuracy drops drastically, then the model is said to be over-fitting or a model that has
# high variance. The key reason for over-fitting is using higher-order polynomial degree
# (may not be required), which will fit decision boundary tools well to all data points
# including the noise of train dataset, instead of the underlying relationship. This will lead
# to a high accuracy (actual vs. predicted) in the train dataset and when applied to the test
# dataset, the prediction error will be high.

# To solve the over-fitting issue:
# • Try to reduce the number of features, that is, keep only the
#    meaningful features or try regularization methods that will keep
#   all the features, however reduce the magnitude of the feature
#   parameter.
# • Dimension reduction can eliminate noisy features, in turn,
#   reducing the model variance.
# • Brining more data points to make training dataset large will also
#   reduce variance.
# • Choosing right model parameters can help to reduce the bias and
#   variance, for example.
#   • Using right regularization parameters can decrease variance
#       in regression-based models.
#   • For a decision tree reducing the depth of the decision tree
#    will reduce the variance.


