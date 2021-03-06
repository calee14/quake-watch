# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
import pickle
# Getting the training data
with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
# Getting the testing data
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# training and finding accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

# finding mean squared error
from sklearn.metrics import mean_squared_error
y_pred = reg.predict(X_test)
reg_mse = mean_squared_error(y_test, y_pred)
reg_rmse = np.sqrt(reg_mse)
print(reg_rmse)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(reg, filename)

# from sklearn.model_selection import GridSearchCV

# parameters = {'n_estimators':[10, 20, 50, 100, 200, 500]}

# grid_obj = GridSearchCV(reg, parameters)
# grid_fit = grid_obj.fit(X_train, y_train)
# best_fit = grid_fit.best_estimator_
# print(best_fit)
# best_fit.predict(X_test)

# best_fit.score(X_test, y_test)