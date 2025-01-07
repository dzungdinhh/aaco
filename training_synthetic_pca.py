import numpy as np
import os
import sys

X_train_digits = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/train_digits_images.npy')
X_train_counter = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/train_counter_images.npy')
y_train = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/train_labels.npy')
X_test_digits = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/test_digits_images.npy')
X_test_counter = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/test_counter_images.npy')
y_test = np.load('/work/users/d/d/ddinh/aaco/input_data/pca_digits/test_labels.npy')

y_train = np.eye(np.max(y_train) + 1)[y_train]
y_test = np.eye(np.max(y_test) + 1)[y_test]

X_train = np.concatenate([X_train_digits, X_train_counter], axis=1)
X_test = np.concatenate([X_test_digits, X_test_counter], axis=1)

# xgb 
masksper = 512
d = X_train.shape[1]
X_class = np.concatenate([X_train]*masksper, 0)
Y_class = np.concatenate([y_train]*masksper, 0)
B = np.concatenate(
[np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(d)], 1, keepdims=True) for _ in range(X_class.shape[0])],
1)
B = np.float32(B.T)
X_class = (X_class[:,:,]*B[:,:,None]).reshape(X_class.shape[0], -1)
X_class = np.concatenate((X_class, B), 1)


# Train classifier
from xgboost import XGBClassifier
est = XGBClassifier(n_estimators=256, device='gpu')
est.fit(X_class, Y_class)


# save model to path
path = '/work/users/d/d/ddinh/aaco/models/'
est.save_model(path + 'synthetic_pca.model')
# est.save_model(path + 'synthetic_raw.model')


Bval = np.concatenate(
  [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(d)], 1, keepdims=True) for _ in range(X_test.shape[0])],
  1)
Bval = np.float32(Bval.T)
Xvalmasked = (X_test[:,:,]*Bval[:,:,None]).reshape(X_test.shape[0], -1)
Xvalmasked = np.concatenate((Xvalmasked, Bval), 1)

val_preds = est.predict_proba(Xvalmasked)

print("accuracy: ", np.mean(np.round(val_preds)==y_test))

# mean number of masks Bval
print("num masks: ", np.mean(np.sum(Bval, axis=1)))