from __future__ import absolute_import, division, print_function
import numpy as np



def read_1m_particular_dataset(filename, sep="\t"):

    from pandas.compat import FileNotFoundError
    try:
        file = open( filename , "r")
    except FileNotFoundError:
        print("Retrying........wait")
        file = open("../" + filename , "r")

    data = np.zeros((6040,3952))

    for row in file:
        row = row.strip()
        u, i, r, _ = list(map(int , row.split(sep)))
        data[u - 1][i - 1] = r
    file.close()


    # for i in range(1 , 6):
    #     mask  = np.where( data > i )
    #     print("Rating : " + str(i) + " count +" + str(  np.sum( mask ) ) )
    #

    return data , data


def read_particular_dataset(filename, sep="\t"):

    from pandas.compat import FileNotFoundError
    try:
        file = open( filename , "r")
    except FileNotFoundError:
        print("Retrying........wait")
        file = open("../" + filename , "r")

    data = np.zeros((943,1682))
    mask = np.zeros( (943 , 1682) )

    for row in file:
        row = row.strip()
        u, i, r, _ = list(map(int , row.split(sep)))
        data[u - 1][i - 1] = r
        mask[u - 1][i - 1] = 1.0
    file.close()


    # for i in range(1 , 6):
    #     mask  = np.where( data > i )
    #     print("Rating : " + str(i) + " count +" + str(  np.sum( mask ) ) )
    #

    return data , mask


def fetch_data_100k_ratings(testIndex):
    data_trainSet , mask_trainSet = read_particular_dataset("data//ml-100k/u" + testIndex + ".base", sep="\t")
    ##data_trainSet = read_particular_dataset("data//ml-100k/u.data", sep="\t")

    data_testSet, _ = read_particular_dataset("data/ml-100k/u" + testIndex + ".test", sep="\t")

    return data_trainSet, data_testSet , mask_trainSet



def fetch_data_1_MILLION_ratings(testIndex):
    data_trainSet , mask_trainSet = read_1m_particular_dataset("data//ml-1m/r" + testIndex + ".train", sep='::')
    ##data_trainSet = read_particular_dataset("data//ml-100k/u.data", sep="\t")

    data_testSet, _ = read_1m_particular_dataset("data//ml-1m/r" + testIndex + ".test", sep='::')

    return data_trainSet, data_testSet , mask_trainSet




def fetch_entire_100k(doNothin):
    testIndex = "1"
    data_trainSet, mask_trainSet = read_particular_dataset("data//ml-100k/u.data", sep="\t")
    data_testSet, _ = read_particular_dataset("data/ml-100k/u" + testIndex + ".test", sep="\t")
    return data_trainSet, data_testSet, mask_trainSet


def fetch_data_entire_1M_ratings(sep="::"):
    file = open('data//ml-1m/ratings.dat', "r")
    data = np.zeros((6040,3952))

    for row in file:
        u, i, r, _ = list(map(int , row.split(sep=sep)))
        data[u - 1][i - 1] = r
    file.close()

    return data


def fetch_data_1M_string(sep="::"):
    file = open('data//ml-1m/ratings.dat', "r")
    data = []


    for row in file:
        u, i, r, _ = list(map(int, row.split(sep=sep)))
        data.append( [ u , i , r] )
    file.close()




def fetch_data_1M_matrix_from_listOlist(listOlist, sep="::"):
    data = np.zeros((6040, 3952))

    for row in listOlist:
        u, i, r = row
        data[u - 1][i - 1] = r


    return data


def fetch_data_1M_counts():
    countList = []
    with open('data/ml-1m/users.dat') as file:
        lines = file.readlines();
        countList.append(len(lines))

    with open('data/ml-1m/movies.dat' , encoding='ISO-8859-1') as file:
        lines = file.readlines();
        countList.append(len(lines) + 69)

    return countList

def fetch_data_100K_counts():
    countList = []
    usersCount = 0
    itemsCount = 0
    with open('data/ml-100k/u.info') as file:
        lines = file.readlines()
        usersCount = lines[0].split(" ")[0]
        itemsCount = lines[1].split(" ")[0]
        file.close()

    countList.append( int(usersCount))
    countList.append(int(itemsCount))
    return countList

def dump_error(filename , x ):
    errorFile = open( filename , "a")
    errorFile.write(str(x) + "\n")
    errorFile.close()


def fill_missing_value(iData, iFilled, initialData, tobeReplaced=0.0):
    missing_mask = np.where(initialData == tobeReplaced)

    # data[missing_mask] = (np.mean(data[missing_mask], axis=0, keepdims=True) / 2)\
    #                      + (np.mean(filler[missing_mask], axis=0, keepdims=True) / 2)
    #
    iData[missing_mask] = iFilled[missing_mask]
    print( " Filling " + str(tobeReplaced) +" " + str(np.sum(iFilled[missing_mask])))
    return iData


if __name__ == "__main__":
    fetch_data_100k_ratings("1")




