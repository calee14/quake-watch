from tensorflow.keras.models import load_model
import pickle
import datetime
import time
import numpy as np
import pandas as pd

# Getting back the objects:
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

X_test = np.load('X_test.npy')
y_test = np.load('n_test.npy')

model = load_model('eq.h5')
model.load_weights('eq_weights.h5')
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

date = '10/17/2019 5:4:15'
ts = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
timestamp = time.mktime(ts.timetuple())
columns = ['Timestamp', 'Latitude', 'Longitude']
data = np.array([[timestamp, 37.7749, 122.4194]]).T
# evaluate_data = pd.DataFrame(data, columns=columns)
# final_data = evaluate_data[['Timestamp', 'Latitude', 'Longitude']]
results = model.predict(X_test)
print(results)
