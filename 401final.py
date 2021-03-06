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
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from data_util import *
from sklearn import svm 
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

data =  pd.read_csv('./data/Salaries.csv')
df = pd.DataFrame(data, columns = ['rank', 'discipline', 'yrs.since.phd', 'yrs.service', 'sex', 'salary'])
#giving the data set seperate cols for females and seperate cols for males
sex = pd.get_dummies(df['sex'])
new = pd.concat([df, sex], axis=1)


features = list(new)

features.remove('salary')
features.remove('rank')
features.remove('discipline')
features.remove('sex')

data_x = new[features]
data_y = new['salary']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)


model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
#      Actual      Predicted
# 283  155865  132415.105887
# 361  109646  107589.019278
# 336   98053  129150.253383
# 64    68404   95500.606669
# 6    175000  121915.778571
# 373  136660  120235.164003
# 102  153303  111742.936505
# 153  103994   94474.198972
# 382   86895  100167.776806
# 70   126320  114870.446036
# 55    83900  108849.146581
# 353  138000  112021.233440
# 396   81035  100727.981662
# 293  104800  106749.281118
# 11    79800  100401.496412
# 167  130664  106793.858446
# 120  115313  103807.302877
# 139  152664  111694.748189
# 216  146000  106934.812407
# 200   92700   94940.401813
# 381  172505  117575.093684
# 14   104800  110247.853227
# 302  170500  124624.037206
# 124   96614  113794.613653
# 300   88600  127655.170104
# 271  194800  142079.542400
# 72   100131  122709.703033
# 347  128250  129335.784672
# 68   111512   97787.239792
# 83    88825   98954.601450
# ..      ...            ...

 print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
     ...: median_absolute_error(y_test, preds), 
     ...: r2_score(y_test, preds),
     ...: explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [715433702.27623689, 20859.891470482231, 0.21227970848099043, 0.22737164083309114]

print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

# Accuracy: 0.0125

# Confusion Matrix:
# [[0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  ..., 
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]]



print('\n ----------------------Part 1: Cross Validation with SVM ---------------------')
mod = svm.SVC(C=2.5)
k_fold = KFold(n_splits=5, shuffle=True, random_state=4)
k_fold_scores = cross_val_score(mod, data_x, data_y, scoring='f1_macro', cv=k_fold)
print('CV Scores (K-Fold): ' + str(k_fold_scores))
#CV Scores (K-Fold): [ 0.01190476  0.00291971  0.0075188   0.00497512  0.        ]

loo = LeaveOneOut() 
loo_scores = cross_val_score(mod, data_x, data_y, cv=loo)
print('CV Scores (Avg. of Leave-One-Out): ' + str(loo_scores.mean()))
#CV Scores (Avg. of Leave-One-Out): 0.0176322418136

shuffle_split = ShuffleSplit(test_size=0.2, train_size=0.8, n_splits=5)
ss_scores = cross_val_score(mod, data_x, data_y, scoring='accuracy', cv=shuffle_split)
print('CV Scores (Shuffle-Split): ' + str(ss_scores))
#CV Scores (Shuffle-Split): [ 0.      0.025   0.0125  0.025   0.0125]

print('\n--------------- PART 2: Grid Search + Cross Validation with RF ----------------')
# Optimize a RF classifier and test with grid search.

param_grid = {'n_estimators':[5, 10, 50, 100], 'max_depth':[3, 6, None]} 

# Find the best RF and use that. Do a 3-fold CV and score with f1 macro.
optimized_rf = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, cv=3, scoring='f1_macro') 
optimized_rf.fit(x_train, y_train) # Fit the optimized RF just like it is a single model

print('Grid Search Test Score (Random Forest): ' + str(optimized_rf.score(x_test, y_test)))
#Grid Search Test Score (Random Forest): 0.0127226463104


print('\n-------------------- PART 3: Model ensemble illustrations --------------------------')

bagging_mod = ensemble.BaggingClassifier(linear_model.LogisticRegression(), n_estimators=200)
k_fold = KFold(n_splits=5, shuffle=True, random_state=4) # Shuffling is needed since classes are ordered.
bagging_mod_scores = cross_val_score(bagging_mod, data_x, data_y, scoring='f1_macro', cv=k_fold)

print('CV Scores (Bagging NB) ' + str(bagging_mod_scores))
#CV Scores (Bagging NB) [ 0.00430108  0.          0.          0.00643382  0.        ]

# Here is a basic voting classifier with CV and Grid Search.
m1 = svm.SVC()
m2 = ensemble.RandomForestClassifier()
m3 = naive_bayes.GaussianNB()
voting_mod = ensemble.VotingClassifier(estimators=[('svm', m1), ('rf', m2), ('nb', m3)], voting='hard')

# Set up params for combined Grid Search on the voting model. Notice the convention for specifying 
# parameters foreach of the different models.
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}
 best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=3)
 best_voting_mod.fit(x_train, y_train)

 print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test, y_test)))
 #Voting Ensemble Model Test Score: 0.025




#------------------------------------------------------------------------------------------------------
 freedman =  pd.read_csv('./data/Freedman.csv')

#there are alot of NAN so I decided to take them out
freedman = freedman.fillna(method ='ffill')

#making the baseline model
data_x = freedman[['population','nonwhite', 'density']]
data_y = freedman[['crime']]
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
#R2 is negative only because chosen model does not follow the trend of the data
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
    ...: median_absolute_error(y_test, preds), 
    ...: r2_score(y_test, preds),
    ...: explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [1350469.8871711888, 712.69331662228842, -0.20270274643912134, -0.20001700896834906]



data_x = freedman[['nonwhite']]
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
   ...: median_absolute_error(y_test, preds), 
   ...: r2_score(y_test, preds),
   ...: explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [1169087.3890582093, 568.66804120302095, -0.041166950114611245, -0.041126772031453429]


data_x = freedman[['population']]
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
    ...: median_absolute_error(y_test, preds), 
    ...: r2_score(y_test, preds),
    ...: explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [1263051.6619337648, 617.85212580903817, -0.12484974091812084, -0.12308289217410318]

data_x = freedman[['density']]
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
    ...: median_absolute_error(y_test, preds), 
    ...: r2_score(y_test, preds),
    ...: explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [1118226.4440938304, 508.62005563876323, 0.0041288382448616545, 0.0047292945548844667]


#getting all predictors
data_x = freedman[list(freedman)[1:4]]
#gettinthe response variable
data_y = freedman[list(freedman)[4]] 
model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
model.fit(x_train,y_train)
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
#      Actual    Predicted
# 16     1434  2355.839752
# 2      5018  2402.941622
# 24     3077  2304.500219
# 93     2130  2235.994342
# 26     1877  2265.509454
# 74     1753  3988.757570
# 5      2805  3131.353195
# 88     5441  3339.008207
# 20     1780  2798.125555
# 19     2680  4559.609169
# 107    2874  2301.013238
# 13     2171  2673.667512
# 84     2977  2287.894433
# 106    2393  2568.651516
# 10     2285  3229.428368
# 71     2560  2680.658853
# 62     2433  3087.067813
# 35     3688  2718.939206
# 25     3701  2964.718934
# 89     3000  2469.543461
# 34     3164  2493.249801
# 53      627  2219.669165


