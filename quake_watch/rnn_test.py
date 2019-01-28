import os; os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

def load_earthquake_data(path='data/'):
    csv_path = os.path.join(path, 'database.csv')
    return pd.read_csv(csv_path)

train_df = load_earthquake_data()
print(train_df.shape)

# get the features we want to pass into the model
train_df = train_df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

import datetime
import time

timestamp = []
for d, t in zip(train_df['Date'], train_df['Time']):
    try:
        # changing the date time features into numeric values
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except:
        # print('ValueError')
        timestamp.append('ValueError')

timeStamp = pd.Series(timestamp)

# add the timestamp values into the feature set
train_df['Timestamp'] = timeStamp.values
final_data = train_df.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != 'ValueError']

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
final_data = sc.fit_transform(final_data)

X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

X = X.values
y = y.values

X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# y = np.reshape(y, (y.shape[0], y.shape[1], 1))

# split the data into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.reset_states()

model.add(LSTM(16, return_sequences=False, input_shape=(3, 1)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='SGD', loss='squared_hinge', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=3, epochs=10, verbose=1, validation_data=(X_test, y_test))

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

results = model.predict(X_test)
results = sc.inverse_transform(results)
print(results)
