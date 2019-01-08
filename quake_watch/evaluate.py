from tensorflow.keras.models import load_model
import pickle
import datetime
import time
import numpy as np

# Getting back the objects:
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

model = load_model('earthquake.h5')
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

date = '10/17/1989 0:0:0'
ts = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
timestamp = time.mktime(ts.timetuple())
a = np.array([timestamp, 37.7749, 122.4194])
print(a)
print(a.shape)
model.predict(np.array([timestamp, 37.7749, 122.4194]))
