# one-hot encoding numpy array
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
labels_vecs = lb.fit_transform(labels)


# split datasets
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

splitter = sss.split(codes, labels_vecs)
train_idx, val_test_idx = next(splitter)
split_size = len(val_test_idx) // 2
val_idx = val_test_idx[:split_size]
test_idx = val_test_idx[split_size:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y =  codes[test_idx], labels_vecs[test_idx]