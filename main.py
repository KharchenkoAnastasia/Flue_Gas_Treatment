import pandas as pd
import Preprocessing
from Preprocessing import *
import numpy as np
from sklearn.model_selection import train_test_split
from LinerRegr import *
from KNeighborsRegressor import *
from RandomForest import *
from SVR import *
from mpl import *
from sklearn.preprocessing import MinMaxScaler   

xl = pd.ExcelFile('C:/Users/kharc/OneDrive/Рабочий стол/doc1.xlsx')
df = xl.parse('Paper', header=None)
df[0].fillna(method='ffill', inplace=True)
Preprocessing.rem_blank(df)
#df=df.iloc[:,[2,3,8,9,11,12,13]].values
#df=df.iloc[:,[2,3,11,12,13]].values
df=df.iloc[:,[2,3,8,9,11,12,13]].values 
df=Preprocessing.fit_missimg_val(df) #filling in missing values
df=Preprocessing.rem_duplicate(df) # вelete duplicates
df=Preprocessing.rem_outliers(df) # removal of outliers

X=np.column_stack(((df[:,0],df[:,2:7])))
y=np.reshape(df[:,1], (-1,1))

#normalization              
scaler_x = MinMaxScaler().fit(X)
scaler_y = MinMaxScaler().fit(y)
X=scaler_x.transform(X)
y=scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# xnew - parameters of the flue gas cleaning process
# xnew[0] - SO2 content before cleaning, 
# xnew[1] - gas volume for the 1st stage, 
# xnew[2] - gas volume for the 2nd stage, 
# xnew[3] - water flow in the adsorber,
# xnew[4] -lime consumption in the adsorber
# xnew[5] - secondary reagent consumption
xnew=np.array([[342., 410., 323.,   1.,   0.,   0.]])
xnew=scaler_x.transform(xnew)

# ynew - SO2 content in gas after cleaning process, ynew=410
ynew=0 

print('///////////////////////////////////////////////////////////')
#Predicting with Linear Regression
print('Linear Regression')
LinearRegression=LinearRegr()
LinearRegression.train(X_train, X_test, y_train, y_test )
ynew=LinearRegression.predict(xnew)
print("X=%s, Predicted=%s" % (scaler_x.inverse_transform(xnew)[0], scaler_y.inverse_transform(ynew)[0]))
print('///////////////////////////////////////////////////////////')
print()

#Predicting with Random Forest
print('Random Forest')
RandomForest=RandomForest()
RandomForest.train(X_train, X_test, y_train, y_test )
ynew=np.reshape(RandomForest.predict(xnew), (-1,1))
print("X=%s, Predicted=%s" % (scaler_x.inverse_transform(xnew)[0], scaler_y.inverse_transform(ynew)[0]))
print('///////////////////////////////////////////////////////////')
print()

# Nearest Neighbor Prediction
print('K-Nearest Neighbors')
KNeighborsRegressor=KNeigRegressor()
KNeighborsRegressor.train(X_train, X_test, y_train, y_test )
ynew=np.reshape(KNeighborsRegressor.predict(xnew), (-1,1))
print("X=%s, Predicted=%s" % (scaler_x.inverse_transform(xnew)[0], scaler_y.inverse_transform(ynew)[0]))
print()

#Prediction with a Neural Network
print('///////////////////////////////////////////////////////////')
print('Neural Network')
mpl=mpl()
mpl.train(X_train, X_test, y_train, y_test,200)
ynew=np.reshape(mpl.predict(xnew), (-1,1))
print("X=%s, Predicted=%s" % (scaler_x.inverse_transform(xnew)[0], scaler_y.inverse_transform(ynew)[0]))
print()




