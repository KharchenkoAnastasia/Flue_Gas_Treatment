from sklearn.linear_model import LinearRegression
#from sklearn import metrics
from print_result import *
import numpy as np
class LinearRegr():
    
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, X_test, y_train, y_test ):
        self.model.fit(X_train, y_train)
        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred = self.model.predict(X_test)
        print_result (y_train, self.y_train_pred,y_test,self. y_test_pred)
       
    
    def predict(self, predict):
        
        y_pred = self.model.predict(predict)
        return y_pred