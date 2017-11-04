from keras import losses
from keras.models import Sequential

import keras
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler

import dataset_manager
from lib.data_utils import convert_non_one_hot
from lib.feature_convertor import read_data_100k_gen_dataset

import numpy as np
from keras.layers import Dense, Activation, initializers





tot_mae = []
tot_acc = []


for i in ["1" , "2" , "3" , "4" , "5"]:

    model = Sequential()

    X_train, y_train, X_test, y_test = read_data_100k_gen_dataset(i);


    model_scalar = StandardScaler();
    model_scalar.fit(X_train)


    X_train = model_scalar.transform(X_train )
    X_test = model_scalar.transform(X_test )


    initVar = initializers.RandomUniform()


    inputDimen = np.shape(X_train)[1]
    outputDimen = 5;

    model.add(Dense(units=1000 ,input_dim=inputDimen, trainable=False
                    , kernel_initializer=initVar))
    model.add(Activation('linear'))

    model.add(Dense(units=2000) )
    model.add(Activation('relu'))



    model.add(Dense(units=outputDimen))
    model.add(Activation('softmax'))



    lossFunc = losses.categorical_crossentropy;

    model.compile(loss=lossFunc,
                  optimizer='Adam' , metrics=['accuracy'] )


    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(X_train, y_train , epochs=20, batch_size=800)

    ##joblib.dump(model , "ELM_Model");


    y_pred = np.argmax(model.predict(X_test) , axis=1)

    y_test = convert_non_one_hot(y_test )

    stats = "Acc" + str(accuracy_score(y_test, y_pred));
    print(stats)

    dataset_manager.dump_error("ELM" , stats)

    stats = "MAE" + str(mean_absolute_error(y_test , y_pred))

    dataset_manager.dump_error("ELM", stats)
