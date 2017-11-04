
from sklearn import linear_model
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np

import dataset_manager
from lib import feature_convertor
from lib.data_utils import convert_non_one_hot
from src.algo.recon import NormalRecon


tot_mae = []
tot_acc = []


for i in ["1" , "2" , "3" , "4" , "5"]:

    X_train, y_train , X_test, y_test = feature_convertor.read_data_100k_gen_dataset(i);

    y_test = convert_non_one_hot(y_test  ,True);

    X_test = X_test[0 : 1000]
    y_test = y_test[0 : 1000]

    X_train = X_train[0 : 8000]
    y_train = y_train[0 : 8000]

    print("Test Size", np.shape(y_test))

    clf = NormalRecon()

    print(clf)

    clf.fit( X_train , y_train)

    y_pred = clf.predict(X_test);

    stats = "Acc" + str(accuracy_score(y_test, y_pred));
    print(stats)

    dataset_manager.dump_error("SRC" , stats)

    stats = "MAE" + str(mean_absolute_error(y_test , y_pred))

    dataset_manager.dump_error("SRC", stats)
