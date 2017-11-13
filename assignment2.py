# Doing importations so I can use approapriate libraries
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

setA =  pd.read_csv('./data/AmesHousingSetA.csv')
#First I need to take a look at this data 
setA.head()

#there are alot of NAN so I decided to take them out
setA = setA.fillna(method ='ffill')

#there are alot of columns with category variables, I need to filter them out
out= pd.DataFrame({'one':[1,2,3] ,'two':[1.0, 2.0, 3.0]})
newcols = filter(lambda i: not(setA[i].dtype in [out['one'].dtype, out['two'].dtype]), list(setA))


# I want to do some exploration on what relations the sale price has to do 
#with the year built of the house   
pt1 = sns.lmplot(x="Year.Built", y="SalePrice", data=setA, order=1, ci=None, scatter_kws={"s": 80})
plt.show()

#now constructing baseline model
nsetA = pd.get_dummies(setA, columns=newcols)
allminusprice = list(nsetA)
allminusprice.remove('SalePrice')
data_x = nsetA[allminusprice]
data_y = setA[['SalePrice']]
# Create a least squares linear regression model.
model = linear_model.LinearRegression()
# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
#fit the model
model.fit(x_train,y_train)
# Make predictions on test data and look at the results.
preds = model.predict(x_test)
 print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
    median_absolute_error(y_test, preds), \
    r2_score(y_test, preds), \
    explained_variance_score(y_test, preds)]))
#results wereMSE, MAE, R^2, EVS: [702330378.63518739, 11316.784565090667, 0.89639362081251672, 0.89717174438047131]


# Use k-best feature selection to build the model --
# Create a top-k feature selector based on the F-scores. Get top 25% best features by F-test.
selector_f = SelectKBest(f_regression, k=3)
selector_f.fit(x_train, y_train)

# Get the columns of the best 25% features. 
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Fit the model.
model.fit(xt_train, y_train)

# Make predictions on test data and look at the results.
preds = model.predict(xt_test)

print('MSE, MAE, R^2, EVS (Top 3 Model): ' + 
    str([mean_squared_error(y_test, preds),
    median_absolute_error(y_test, preds),
    r2_score(y_test, preds),
    explained_variance_score(y_test, preds)]))   





#I was also intrested intrested in the year it was remodeled and the sale price
pt2 = sns.lmplot(x="Year.Remod.Add", y="SalePrice", data=setA, order=1, ci=None, scatter_kws={"s": 80})


data_x = setA['Year.Built'] 
data_x2 = setA[['Year.Remod.Add']]
data_y = setA[['SalePrice']]


# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
x_train, x_test, y_train, y_test = train_test_split(data_x2, data_y, test_size = 0.2, random_state = 4)

linear_mod = linear_model.LinearRegression()

# Fit the model.
linear_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = linear_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
	median_absolute_error(y_test, preds), \
	r2_score(y_test, preds), \
    explained_variance_score(y_test, preds)]))

#results are MSE, MAE, R^2, EVS: [702.84885348573698, 17.489290420316593, 0.26331506241265323, 0.26530621433338009]


# now I tried another model with year build and year remodeled

 data_x2 = setA[['Year.Remod.Add', 'Year.Built']]

x_train, x_test, y_train, y_test = train_test_split(data_x2, data_y, test_size = 0.2, random_state = 4)

 linear_mod = linear_model.LinearRegression()

linear_mod.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

preds = linear_mod.predict(x_test)

print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
    ...: ^Imedian_absolute_error(y_test, preds), \
    ...: ^Ir2_score(y_test, preds), \
    ...:     explained_variance_score(y_test, preds)]))
# results were MSE, MAE, R^2, EVS: [4534359459.0967245, 33456.518088229932, 0.33110032004531897, 0.33407607849014187]


#now trying it for year build and year remodeled and total rooms above garage
 data_x2 = setA[['Year.Remod.Add', 'Year.Built', 'TotRms.AbvGrd']]

x_train, x_test, y_train, y_test = train_test_split(data_x2, data_y, test_size = 0.2,
    random_state = 4)

linear_mod = linear_model.LinearRegression()

linear_mod.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

preds = linear_mod.predict(x_test)

In [35]: print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
     ^Imedian_absolute_error(y_test, preds), \
     ^Ir2_score(y_test, preds), \
      explained_variance_score(y_test, preds)]))
# reults were MSE, MAE, R^2, EVS: [3423324388.603826, 27225.634674115106, 0.49499800168592967, 0.4970689174096935]


#now trying it for year build and year remodeled ,total rooms above garage and Garage.Area

data_x2 = setA[['Year.Remod.Add', 'Year.Built', 'TotRms.AbvGrd', 'Garage.Area']]

 x_train, x_test, y_train, y_test = train_test_split(data_x2, data_y, test_size = 0.2,
  random_state = 4)

 linear_mod = linear_model.LinearRegression()

linear_mod.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

preds = linear_mod.predict(x_test)

 print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
 	^Imedian_absolute_error(y_test, preds), \
    ^Ir2_score(y_test, preds), \
    explained_variance_score(y_test, preds)]))
#results were MSE, MAE, R^2, EVS: [2813225623.8042088, 24804.644494614564, 0.58499855682420832, 0.58754226861385295]

#_____________________________________________________________________________________

#now working with the B dataset

setB =  pd.read_csv('./data/AmesHousingSetB.csv')
setB.head()
nsetB = setB.fillna(method='ffill')
outB= pd.DataFrame({'one':[1,2,3] ,'two':[1.0, 2.0, 3.0]})
newcolsB = filter(lambda i: not(nsetB[i].dtype in [outB['one'].dtype, outB['two'].dtype]), list(nsetB))
nsetB = pd.get_dummies(nsetB, columns= newcolsB)
allminusprice = list(nsetB)
allminusprice.remove('SalePrice')
data_x = nsetB[allminusprice]
data_y = setB[['SalePrice']]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()
linear_mod.fit(x_train,y_train)
preds = linear_mod.predict(x_test)

print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
	median_absolute_error(y_test, preds), 
	r2_score(y_test, preds), 
    explained_variance_score(y_test, preds)]))
#MSE, MAE, R^2, EVS: [806714778.34133482, 15857.763224527996, 0.84252688690793653, 0.84380781985198783]

#using Lasso regression to handle cases with correlated predictors
base_mod = linear_model.LinearRegression()

base_mod.fit(x_train,y_train)

preds = base_mod.predict(x_test)

print('R^2 : ' + str(r2_score(y_test, preds)))
#results were: R^2 : 0.842526886908

# Show Lasso regression fits for different alphas.
alphas = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

 for a in alphas:
 		 # Normalizing transforms all variables to number of standard deviations away from mean.
         lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
         lasso_mod.fit(x_train, y_train)
         preds = lasso_mod.predict(x_test)
         print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))

# R^2 (Lasso Model with alpha=0.0): 0.844811695256
# R^2 (Lasso Model with alpha=0.01): 0.84522881459
# R^2 (Lasso Model with alpha=0.1): 0.848006431821
# R^2 (Lasso Model with alpha=0.25): 0.852592347739
# R^2 (Lasso Model with alpha=0.5): 0.859068715356
# R^2 (Lasso Model with alpha=1.0): 0.868597594276
# R^2 (Lasso Model with alpha=2.5): 0.885206384282
# R^2 (Lasso Model with alpha=5.0): 0.899070539579

