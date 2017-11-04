import numpy as np
from pandas.compat import FileNotFoundError
from sklearn.externals import joblib
import os

OCCUPATION_COUNT = 21;

FOLD_TRAIN_SIZE = 80000
FOLD_TEST_SIZE = 20000
RATING_SIZE = 5;


MAX_AGE_BIN_LENGTH = len(bin(73)[2:]);
GENERE_ONEHOT =19

def getOneHotOcc(id):
    vec = [0] * OCCUPATION_COUNT;##  np.zeros( (OCCUPATION_COUNT,1 ) );
    vec[id - 1] = 1
    return vec


def getOccupationDict():
    filename ="data//ml-100k/u" + ".occupation";
    try:
        file = open(filename, "r")
    except FileNotFoundError:
        print("Retrying........wait")
        file = open("../" + filename, "r")

    index = 1;

    oocDict = {}
    for row in file:
        row = row.strip();
        occ =  row
        oocDict[occ] = getOneHotOcc(index);
        index = index + 1;

    ##oocDict.items();

    return oocDict;




def readUserRowsAsFeatuers(filename , sep, genderDict , occDict):
    from pandas.compat import FileNotFoundError
    try:
        file = open(filename, "r")
    except FileNotFoundError:
        print("Retrying........wait")
        file = open("../" + filename, "r")


    mAge = -1;
    minAge = 1000;
    userDict = dict()
    for row in file:
        row = row.strip()
        l = row.split(sep);

        l[2] = genderDict[l[2]]
        occup = occDict[l[3]]
        l = l[:3]

        id , age, gender  = list(map(int, l))

        age = list(map(int, bin(age)[2:].zfill( MAX_AGE_BIN_LENGTH ) ));

        if(gender == 1):
            gender = [0 , 1]
        else:
            gender = [1 , 0];

        # mAge = max( [mAge , age]);
        # minAge = min([minAge, age]);
        #
        feat =  age + gender + occup ;
        userDict[id] =  feat;

    file.close()

    return userDict;



def dump_user_item_dict():
    genderDict = {"M": 1, "F": 0};
    occDict = getOccupationDict();

    userDict = readUserRowsAsFeatuers("data//ml-100k/u" + ".user", "|", genderDict, occDict)

    itemDict = readItemRowsAsFeatuers("data//ml-100k/u" + ".item", "|")

    joblib.dump(userDict , "userDict")
    joblib.dump(itemDict, "itemDict")




def one_hot_data(filename , sep , userDict ,  genderDict , occDict, itemDict):
        from pandas.compat import FileNotFoundError
        try:
            file = open(filename, "r")
        except FileNotFoundError:
            print("Retrying........wait")
            file = open("../" + filename, "r")

        if 'test' in filename:
            localLimit = FOLD_TEST_SIZE
        else:
            localLimit = FOLD_TRAIN_SIZE


        X = np.zeros((localLimit , MAX_AGE_BIN_LENGTH + len(genderDict) + OCCUPATION_COUNT + GENERE_ONEHOT))
        y = np.zeros((localLimit , RATING_SIZE));

        for index, row in enumerate(file):
            row = row.strip()
            u_id, i_id, r, _ = list(map(int, row.split(sep)))
            user = userDict[u_id];
            item = itemDict[i_id];

            iFeature = np.asarray( user + item );
            X[index] = iFeature;
            y[index][r - 1] = 1.0;

        if((index - (localLimit - 1)) != 0):
            raise ValueError("Diff in LIMIT", (index - (localLimit - 1)))

        file.close()

        return X,  y;


def read_data_100k_gen_dataset(index):

    basePath = os.path.dirname(os.path.abspath(__file__)) + "//"
    userDict = joblib.load(basePath + "userDict")
    itemDict = joblib.load(basePath + "itemDict")
    genderDict = {"M": 1, "F": 0};
    occDict = getOccupationDict();

    X_train , y_train = one_hot_data("data//ml-100k/u" + index + ".base", "\t" , userDict ,
                         genderDict , occDict , itemDict)

    X_test, y_test = one_hot_data("data//ml-100k/u" + index + ".test", "\t", userDict,
                        genderDict, occDict, itemDict)

    return X_train , y_train , X_test , y_test;


def readItemRowsAsFeatuers(filename , sep):
    from pandas.compat import FileNotFoundError
    try:
        file = open(filename, "r")
    except FileNotFoundError:
        print("Retrying........wait")
        file = open("../" + filename, "r")


    itemDict = dict()
    for row in file:
        row = row.strip()
        l = row.split(sep);
        id = int(l[0]);
        genere = list(map(int ,  l[-19:] ))
        itemDict[id] = genere;

    file.close()

    return itemDict;

if __name__ == '__main__':
    print("Main should n';t get called")
    #dump_user_item_dict()
    #X_train, y_train, X_test, y_test = read_data_100k_gen_dataset("1");

