from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from keras import optimizers
from matplotlib import pyplot
import numpy as np
#from IPython import get_ipython
from keras.layers import Dense, Flatten, Dropout
#from matplotlib import pyplot
import matplotlib.pyplot as plt

class mpl:
    def __init__(self):
        model = Sequential()
        model.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mse','mae'])
        self.model = model
        
    def train (self,X_train, X_test, y_train, y_test,epochs):
        
        self.history=self.model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=10, verbose=0)
        ##print(history.history.keys())        
        print('MAE on train set:', np.mean(self.history.history['mae']))
        print('MSE on train set:', np.mean(self.history.history['val_mae']))
        print('------------------------------------')        
        print('MAE on test set:', np.mean(self.history.history['mse']))
        print('MSE on test set:', np.mean(self.history.history['val_mse']))

    def plot_train(self):
        
        fig = plt.figure()
        pyplot.plot(self.history.history['mae'], label='train')
        pyplot.plot(self.history.history['val_mae'], label='test')
        pyplot.title('model mean_absolute_error')
        pyplot.ylabel('model mean_absolute_error')
        pyplot.xlabel('epoch')
        pyplot.legend()
        pyplot.show()
         
        fig = plt.figure()
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.title('model mean_square_error')
        pyplot.ylabel('model mean_square_error')
        pyplot.xlabel('epoch')
        pyplot.legend()
        pyplot.show()
        
    def predict(self, xnew):
        ynew = self.model.predict(xnew)
        return ynew

