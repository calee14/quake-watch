import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import pandas as pd

def load_earthquake_data(path='data/'):
    csv_path = os.path.join(path, 'database2.csv')
    return pd.read_csv(csv_path)

earthquakes = load_earthquake_data()
# get the features we want to pass into the model
earthquakes = earthquakes[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

import datetime
import time

timestamp = []
for d, t in zip(earthquakes['Date'], earthquakes['Time']):
    try:
        # changing the date time features into numeric values
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except:
        # print('ValueError')
        timestamp.append('ValueError')

timeStamp = pd.Series(timestamp)
# add the timestamp values into the feature set
earthquakes['Timestamp'] = timeStamp.values
final_data = earthquakes.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != 'ValueError']

X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

from sklearn.model_selection import train_test_split

# split the data into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train.to_pickle('X_train.pkl')
y_train.to_pickle('y_train.pkl')
X_test.to_pickle('X_test.pkl')
y_test.to_pickle('y_test.pkl')
