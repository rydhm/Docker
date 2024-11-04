###################################################################
#### Create Loan Data for Classification in Python ####
import pandas as pd
import numpy as np
ColumnNames=['CIBIL','AGE', 'SALARY', 'APPROVE_LOAN']
DataValues=[[480, 28, 610000, 'Yes'],
             [480, 42, 140000, 'No'],
             [480, 29, 420000, 'No'],
             [490, 30, 420000, 'No'],
             [500, 27, 420000, 'No'],
             [510, 34, 190000, 'No'],
             [550, 24, 330000, 'Yes'],
             [560, 34, 160000, 'Yes'],
             [560, 25, 300000, 'Yes'],
             [570, 34, 450000, 'Yes'],
             [590, 30, 140000, 'Yes'],
             [600, 33, 600000, 'Yes'],
             [600, 22, 400000, 'Yes'],
             [600, 25, 490000, 'Yes'],
             [610, 32, 120000, 'Yes'],
             [630, 29, 360000, 'Yes'],
             [630, 30, 480000, 'Yes'],
             [660, 29, 460000, 'Yes'],
             [700, 32, 470000, 'Yes'],
             [740, 28, 400000, 'Yes']]

#Create the Data Frame
LoanData=pd.DataFrame(data=DataValues,columns=ColumnNames)
LoanData.head()

#Separate Target Variable and Predictor Variables
TargetVariable='APPROVE_LOAN'
Predictors=['CIBIL','AGE', 'SALARY']
X=LoanData[Predictors].values
y=LoanData[TargetVariable].values

############################################################
# Random Search CV
from sklearn.model_selection import RandomizedSearchCV

#Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()

# Parameters to try
Parameter_Trials={'n_estimators':[100,200,300,500,1000],
                  'criterion':['gini','entropy'],
                  'max_depth': [2,3]}

Random_Search = RandomizedSearchCV(RF, Parameter_Trials, n_iter=5, cv=5, n_jobs=1, verbose=5)
RandomSearchResults=Random_Search.fit(X,y)