from tensorflow.keras.models import load_model
import pickle
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# Getting back the objects:
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

filename = 'finalized_model.sav'

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)

date = '10/17/1987 5:4:15'
ts = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
timestamp = time.mktime(ts.timetuple())
columns = ['Timestamp', 'Latitude', 'Longitude']
data = np.array([[timestamp, 37.7749, 122.4194], [timestamp, 100.7749, 12.4194], [timestamp, 39.7749, 32.4194]])
data = np.reshape(data, (data.shape[0], data.shape[1], 1))
print(data)
# evaluate_data = pd.DataFrame(data, columns=columns)
# final_data = evaluate_data[['Timestamp', 'Latitude', 'Longitude']]
results = model.predict(data)
print(results)
