import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def dumb_regressor_result(x_test, x_train, y_test, y_train):
    """
          Dumb regressor, predict only the mean value for each target variable,
          returns MAE and MSE metrics per each variable.

          Args:
            x_test: validation samples
            x_train: training samples
            y_test: validation target
            y_train: training target

          Returns:
            dumb_metrics: list of metrics results after dumb regression
    """
    dumb_reg = DummyRegressor()
    fake_data = np.zeros((x_train.shape[0], 1))
    fake_test = np.zeros((1, 1))
    dumb_reg.fit(fake_data, y_train)
    dumb_pred = dumb_reg.predict(fake_test)[0]
    dumb_metrics = []
    for i in range(dumb_pred.size):
        dumb_pred_var = np.full((x_test.shape[0], 1), dumb_pred[i])
        dumb_mse_var = mean_squared_error(y_test[:, i], dumb_pred_var)
        dumb_mae_var = mean_absolute_error(y_test[:, i], dumb_pred_var)
        dumb_metrics.append([dumb_mse_var, dumb_mae_var])
    return dumb_metrics
