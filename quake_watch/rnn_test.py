import os; os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow.keras
import tensorflow.keras.backend as K

def load_earthquake_data(path='data/'):
    csv_path = os.path.join(path, 'database.csv')
    return pd.read_csv(csv_path)

train_df = load_earthquake_data()
print(train_df.shape)
