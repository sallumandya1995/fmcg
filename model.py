#!/usr/bin/env python 
#  
"""
Created on Sun May 10 01:28:58 2020

@author: arsalan
"""

#importing packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
import pickle
train = pd.read_csv("fmcg.csv")
train.head(3)
train.ACH_IN_EA=train['ACH_IN_EA'].apply(lambda x: int(x.replace(',','')))
train.ACH_IN_EA= train.ACH_IN_EA.astype('float')
train.TARGET_IN_EA=train['TARGET_IN_EA'].apply(lambda x: int(x.replace(',','')))
train.TARGET_IN_EA= train.TARGET_IN_EA.astype('float')
data1=train[['ACH_IN_EA', 'TARGET_IN_EA','PLAN_MONTH','SLSMAN_CD','PROD_CD']]
data1.head(3)
#creating dummies for categorical variables

dummies = pd.get_dummies(data1[['PLAN_MONTH','SLSMAN_CD','PROD_CD']])
# Dropping the columns for which we have created dummies
data1.drop(['PLAN_MONTH','SLSMAN_CD','PROD_CD'],inplace=True,axis = 1)
# adding the columns to the  data frame 
data2 = pd.concat([data1,dummies],axis=1)
data2.head(3)
X = data2.iloc[:,1:313]
Y = data2.iloc[:,0]
X.head(3)
Y.head(3)
#split the data to train the model 
X_train,X_test,y_train,y_test = train_test_split(data2,Y,test_size = 0.3,random_state= 0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.head(3)
#Defining cross_val_score function for both train and test sets separately
n_folds = 5
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
scorer = make_scorer(mean_squared_error,greater_is_better = False)
def rmse_CV_train(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(data2.values)
    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
def rmse_CV_test(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(data2.values)
    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
#Linear model without Regularization
lr = LinearRegression()
lr.fit(X_train, y_train)

test_pre = lr.predict(X_test)
train_pre = lr.predict(X_train)
# Look at predictions on training and validation set
print('rmse on train', rmse_CV_train(lr).mean())
print('rmse on train',rmse_CV_test(lr).mean())
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
