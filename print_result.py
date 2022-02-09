from sklearn import metrics
import numpy as np
def print_result(y_train, y_train_pred,y_test, y_test_pred):
    

    print('MAE on train set:', metrics.mean_absolute_error(y_train, y_train_pred))
    print('MSE on train set:', metrics.mean_squared_error(y_train,y_train_pred))
    print('RMSE on train set:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    print('R^2 on train set:', metrics.r2_score(y_train, y_train_pred))
    print('------------------------------------')
    print('MAE on test set:', metrics.mean_absolute_error(y_test, y_test_pred))
    print('MSE on test set:', metrics.mean_squared_error(y_test, y_test_pred))
    print('RMSE on test set:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    print('R^2 on test set:', metrics.r2_score(y_test, y_test_pred))