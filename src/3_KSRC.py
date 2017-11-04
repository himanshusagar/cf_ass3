
from sklearn import linear_model
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np

from lib import feature_convertor
from lib.data_utils import convert_non_one_hot
from src.algo.kernel_recon import KernelRecon

X_train, y_train , X_test, y_test = feature_convertor.read_data_100k_gen_dataset("1");

y_test = convert_non_one_hot(y_test  ,True);

X_test = X_test[0 : 1000]
y_test = y_test[0 : 1000]

X_train = X_train[0 : 10000]
y_train = y_train[0 : 10000]

print("Test Size", np.shape(y_test))

clf = KernelRecon()

print(clf)

clf.fit( X_train , y_train)

y_pred = clf.predict(X_test);

print("Acc")
print(accuracy_score(y_test , y_pred))
print("MAE")
print(mean_absolute_error(y_test , y_pred))