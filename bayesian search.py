# installing library for Bayesian optimization
pip install hyperopt
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

##############################################################
# Bayesian hyperparameter optimization
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
from sklearn.model_selection import cross_val_score

#Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()

# Defining the hyper parameter space as a dictionary
parameter_space = { 'n_estimators': hp.quniform('n_estimators',5,50,5),
                       'max_depth': hp.quniform('max_depth', 2,10,1),
                      'criterion': hp.choice('criterion', ['gini', 'entropy'])
                  }

# Defining a cost function which the Bayesian algorithm will optimize
def objective(parameter_space):
    
    # The accuracy parameter is the average accuracy obtained by cross validation of the data
    # See different scoring methods by using sklearn.metrics.SCORERS.keys()
    Error = cross_val_score(RF, X, y, cv = 5, scoring='accuracy').mean()

    # We return the loss which will be minimized by the fmin() function
    return {'loss': -Error, 'status': STATUS_OK }

import warnings
warnings.filterwarnings('ignore')

# Finding out which set of hyperparameters give highest accuracy
trials = Trials()
best_params = fmin(fn= objective,
            space= parameter_space,
            #algo= tpe.suggest,
            algo=anneal.suggest,  # the logic which chooses next parameter to try
            max_evals = 100,
            trials= trials)