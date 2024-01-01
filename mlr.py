#OBJECTIVE IS TO SUGGEST ONLY one ATTRIBUTE TO BE CONSIDERED FOR BUSINESS 
#AMONG MULTIPLE ATTRIBUTES

#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATASET
data=pd.read_csv(r"C:\Users\desti\Desktop\DATASCIENCE\ML\REGRESSION\MultipleLinearRegression\50_Startups.csv")

#DATA PRE-PROCESSING AND CLEANSING 
#FIRST FIVE ROWS
data.head()

#NO OF ROWS AND COLUMNS
data.shape

#CHECKING FOR MISSING VALUES
data.info()
data.isnull().sum()

#DESCRIPTIVE STATISTICS
data.describe()

#COLUMNS NAMES
data.columns

#SEPERATING DATA INTO DEPENDENT AND INDEPENDENT VARIBALES
x=data.iloc[ : , :-1]
y=data.iloc[ : , 4]

#WE HAVE ONE CATEGORICAL ATTRIBUTE IN INDEPENDENT VARIBALE
#CONVERTING IT INTO NUMERICAL VALUES USING DUMMY VARIBALE TRAP (IMPUTATION)
x=pd.get_dummies(x)

#VISUALIZATION
attribute1=data['R&D Spend']
plt.scatter(attribute1,y,color='Red')
plt.plot(attribute1,y,color='black')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('R&D Spend vs Profit')
plt.legend()
plt.show()

attribute2=data['Administration']
plt.scatter(attribute2,y,color='Red')
plt.plot(attribute2,y,color='black')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.title('Administration vs Profit')
plt.show()

attribute3=data['Marketing Spend']
plt.scatter(attribute3,y,color='Red')
plt.plot(attribute3,y,color='black')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Marketing Spend vs Profit')
plt.show()

attribute4=data['State']
plt.hist(attribute4,color='red')
plt.xlabel('State')
plt.ylabel('Count')
plt.title('State vs Count')
plt.show()

#SPLITTING DATA INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#PREDICITON
y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test, y_pred)
r2

#NOW WE NEED TO CHOOSE THE ONLY ONE ATTRIBUTE WHICH HAS HIGHEST SIGNIFICANCE

import statsmodels.api as sm

#EQUATION OF REGRESSION Y=B0+B1X1+B2X2+B3X3+.... WE DONT HAVE B0 WHICH IS INTERECEPT
#SO NEED TO APPEND IT INTO X 

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) 

#NOW WE START BACKWARD ELIMINATION we will consider 6 attributes 
#out of 7 in order to be prevent from dummy variable trap
x_opt=x[: ,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[: ,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[: ,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[: ,[0,1,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[: ,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

