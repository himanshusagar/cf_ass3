import numpy as np
from sklearn.metrics import mean_absolute_error


def calc_nmae(true , pred):

    tot_mae = 0
    rows = np.shape(true)[0]
    cols= np.shape(true)[1]
    denominator = 0;

    tot_rmse = 0

    for user in range(rows):
        for item in range(cols):
            if(true[user][item] > 0):
                true_rat = true[user][item]
                pre_rat = pred[user][item]
                i_mae = np.absolute(float(true_rat) - float(pre_rat))
                tot_mae = tot_mae + i_mae
                denominator = denominator + 1



    solution = tot_mae/denominator

    print (rows * cols)
    return solution

