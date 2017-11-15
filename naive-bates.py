import pandas as pd
from sklearn.model_selections import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from data_util import *
data = pd.read_csv('./data/iris.csv')
features = list(data)
features.remove('Species')
data_x =data[features]
data_y = data['Species']
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)
data_y
#Out[20]: 
#array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, tes
    t_size=0.3, random_state=4)
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
GaussianNB(priors=None)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)
#Accuracy: 0.977777777778
#Avg. F1 (Micro): 0.977777777778
#Avg. F1 (Macro): 0.971781305115
#Avg. F1 (Weighted): 0.977895355673
#             precision    recall  f1-score   support

#          0       1.00      1.00      1.00        21
#         1       0.91      1.00      0.95        10
#        2       1.00      0.93      0.96        14

#avg / total       0.98      0.98      0.98        45

#Confusion Matrix:
#[[21  0  0]
 #[ 0 10  0]
 #[ 0  1 13]]

y_test

#array([2, 0, 2, 2, 2, 1, 1, 0, 0, 2, 0, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 2, 1,
       0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 1, 0, 2])

preds

#array([2, 0, 2, 2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 2, 1,
       0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 1, 0, 2])

y_test_labs = le.inverse_transform(y_test)

preds_labs = le.inverse_transform(preds)

zip(y_test_labs, preds_labs)
