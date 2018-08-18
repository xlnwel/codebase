# convert categorical data into dummy/indicator data
import pandas as pd

pd.get_dummies(data, prefix)


# split pandas data into training and testing sets
import numpy as np
# assuming data is a pandas dataframe
sample_idx = np.random.choice(data.index, size=int(0.9 * len(data)), replace=False)
train_data, test_data = data.iloc[sample_idx], data.drop(sample_idx)

