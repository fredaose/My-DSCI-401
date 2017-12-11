import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
      

data =  pd.read_csv('./data/employee-perf.csv')
data_x = data[['Aptitude Test Score', 'Interview Score', 'Missed Training Classes']]
data_y = data['Annual Performance Rating']
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
   Actual  Predicted
3      90  88.640209
4      85  81.412110
6      94  93.320892
 print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
    ...: median_absolute_error(y_test, preds), 
    ...: r2_score(y_test, preds), 
    ...: explained_variance_score(y_test, preds)]))
MSE, MAE, R^2, EVS: [5.0610589164729705, 1.3597910272418403, 0.62664319468642016, 0.8861576085020817]


#reading in the new modified employee-perf
 data2=  pd.read_csv('./data/employee-perf2.csv')
 #predicting the performance score for data2 based on the performance score of the first data
 data_x = data[['Aptitude Test Score', 'Interview Score', 'Missed Training Classes']]
 data_y= data['Annual Performance Rating']

 predict_vars = data2[['Aptitude Test Score', 'Interview Score', 'Missed Training Classes']]
 model = linear_model.LinearRegression()
 model.fit(x_train,y_train)
 preds = model.predict(predict_vars)

# the numbers are slightly off here from what I had in my answer document but
# they are around the same range.
pprint.pprint(pd.DataFrame({'Predicted':preds}))
  Predicted
  90.747882
  54.092286
  95.026040

  #now moving on to the churn data
  churn = pd.read_csv('./data/churn_data.csv')
  data_x = churn[['Age', 'FamilySize', 'Education', 'Calls', 'Visits']]
  data_x = pd.get_dummies(data_x)
  data_y = churn['Churn']
  le = preprocessing.LabelEncoder()
  data_y = le.fit_transform(data_y)
   
pt1 = sns.lmplot(x="Age", y="Churn", data=churn, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

pt1 = sns.lmplot(x="FamilySize", y="Churn", data=churn, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

pt1 = sns.lmplot(x="Education", y="Churn", data=churn, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

pt1 = sns.lmplot(x="Calls", y="Churn", data=churn, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

pt1 = sns.lmplot(x="Visits", y="Churn", data=churn, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

model = linear_model.LinearRegression()
# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
    ...: median_absolute_error(y_test, preds),
    ...: r2_score(y_test, preds), 
    ...: explained_variance_score(y_test, preds)])) 

test = pd.read_csv('./data/churn_test.csv')
data_x = test[['Age', 'FamilySize', 'Education', 'Calls', 'Visits']]
data_x = pd.get_dummies(data_x)
data_y = test['Churn']
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
log_mod = linear_model.LogisticRegression()
log_mod.fit(x_train, y_train)

# Make predictions - both class labels and predicted probabilities.
preds = log_mod.predict(x_test)
pred_probs = log_mod.predict_proba(x_test)
prob_pos = pred_probs.transpose()[1]  # P(X = 1) is column 1
prob_neg = pred_probs.transpose()[0]  # P(X = 0) is column 0
print(pred_probs)

# Look at results.
pred_df = pd.DataFrame({'Actual':y_test, 'Predicted Class':preds, 'P(1)':prob_pos, 'P(0)':prob_neg})
print(pred_df.head(15))
print('Accuracy: ' + str(accuracy_score(y_test, preds)))
print('Precison: ' + str(precision_score(y_test, preds)))
print('Recall: ' + str(recall_score(y_test, preds)))
print('F1: ' + str(f1_score(y_test, preds)))
print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
   Actual      P(0)      P(1)  Predicted Class
20       0  0.961589  0.038411                0
15       0  0.957897  0.042103                0
17       0  0.879351  0.120649                0
2        1  0.098803  0.901197                1
11       0  0.465074  0.534926                1
19       0  0.998385  0.001615                0
16       1  0.021083  0.978917                1
27       1  0.882716  0.117284                0
22       0  0.656296  0.343704                0
28       0  0.067305  0.932695                1
Accuracy: 0.7
Precison: 0.5
Recall: 0.666666666667
F1: 0.571428571429
ROC AUC: 0.690476190476
Confusion Matrix:
[[5 2]
 [1 2]]


from sklearn import naive_bayes
from data_util import *

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

Accuracy: 0.9
Avg. F1 (Micro): 0.9
Avg. F1 (Macro): 0.89010989011
Avg. F1 (Weighted): 0.903296703297
             precision    recall  f1-score   support

          0       1.00      0.86      0.92         7
          1       0.75      1.00      0.86         3

avg / total       0.93      0.90      0.90        10

Confusion Matrix:
[[6 1]
 [0 3]]

