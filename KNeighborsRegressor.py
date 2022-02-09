from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import numpy as np
from print_result import *
class KNeigRegressor():
    
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors=5)
        
    def train(self, X_train, X_test, y_train, y_test ):
        self.model.fit(X_train, y_train)
        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred = self.model.predict(X_test)
        print_result (y_train, self.y_train_pred,y_test,self. y_test_pred)
        
    
    def predict(self, predict):
        
        y_pred = self.model.predict(predict)
        return y_pred