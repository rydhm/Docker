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

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###################################################################
#### Trying out different hyperparameters for Random Forest ####
from sklearn.ensemble import RandomForestClassifier

# Change values of n_estimators = 100, 200, 300, 400 etc.
# Change values of criterion = 'gini', 'entropy' etc.
# Change values of max_depth as 3, 4, 5 etc.
clf = RandomForestClassifier(max_depth=3, n_estimators=100,criterion='gini')
 
#Printing all the parameters of Random Forest
print(clf)
 
#Creating the model on Training Data
RF=clf.fit(X_train,y_train)
prediction=RF.predict(X_test)
 
#Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))
 
#Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')
 
#Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults['TargetColumn']=y_test
TestingDataResults['Prediction']=prediction
TestingDataResults.head()