import gc
from scipy import sparse

import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.data_utils import convert_non_one_hot
#
# from pyunlocbox import solvers
# from pyunlocbox import functions

from sklearn import preprocessing

from ext_lib.spgl1.spgl1 import spg_bpdn

import l1ls as L_SOLVER


class NormalRecon:
    def __init__(self):
        self.data_clasees = [-1, 1, 2, 3, 4, 5];
        self.last_indices = [-1, -1, -1, -1, -1, -1];
        self.A_iClass = []

        self.scalar = StandardScaler();

    def fit(self, X, y):
        #
        # self.X = X;
        # self.Y = y;

        self.scalar.fit(X)

        X = self.scalar.transform(X)

        # pca = PCA(n_components=2);
        #
        # pca.fit(X);
        #
        # print("Variance")
        # print(np.sum(pca.explained_variance_ratio_))

        y = convert_non_one_hot(y, True)

        ###Across column
        #Kernel Recon
       # X = preprocessing.normalize(X, norm='l2', axis=0)

        dataXY = np.column_stack((X, y))

        # For taking uNiq
        b = dataXY[np.lexsort(dataXY.reshape((dataXY.shape[0], -1)).T)];
        probUNi = b[np.concatenate(([True], np.any(b[1:] != b[:-1], axis=tuple(range(1, dataXY.ndim)))))]

        print("Old Size" , np.shape(dataXY))
        print("New Size", np.shape(probUNi))
        dataXY = probUNi;


        gc.collect()

        self.m = np.shape(dataXY)[0];
        self.n = np.shape(dataXY)[1];

        ind = np.argsort(dataXY[:, self.n - 1]);
        dataXY = dataXY[ind]

        for i in range(self.m):
            iClass = dataXY[i, self.n - 1];
            self.last_indices[int(iClass)] = i;

        dataXY = np.transpose(dataXY)
        self.A =  dataXY[: self.n - 1, :]

        print("self.last indices")
        print(self.last_indices)

        for iClass in range(1, 6):
            lastGuy = self.last_indices[iClass - 1];
            thisGuy = self.last_indices[iClass];
            self.A_iClass.append( self.A[:, lastGuy + 1: thisGuy + 1] )


    def predict(self, X_test):

        X_test = self.scalar.transform(X_test)

        #X_test = preprocessing.normalize(X_test , norm='l2', axis=0)

        y_predict = np.zeros((np.shape(X_test)[0] , 1))


        A = self.A
        dimen = np.shape(A)[0];
        I = np.eye( dimen )
        P = np.column_stack( (A, I) )

        # error = np.random.normal( 0 , 0.1,
        #                       ( dimen , ) )
        #


        sigma = 0.01  # % Desired ||Ax - b||_2

        print('%s ' % ('-' * 78))
        print('NormalRecons the basis pursuit denoise (BPDN) problem:      ')
        print('                                              ')
        print('  minimize ||x||_1 subject to ||Ax - b||_2 <= ' , sigma)
        print('                                              ')
        print('%s%s ' % ('-' * 78, '\n'))

        LIMIT = len(y_predict);

        print()
        gc.collect();

        sP = sparse.csr_matrix(P);

        for glob_index in range( LIMIT ):
            print("#" + str(glob_index) , " ")

            # % Set up vector b, and run solver
            b = X_test[glob_index];
            # print("b_shape" , np.shape(b));

            yeta =  b;
            # yeta = np.row_stack(( b.reshape( np.shape(b)[0] ,1)
            #                       , error.reshape( np.shape(error)[0] ,1 ) ));
            #
            # yeta = yeta.reshape( (np.shape(yeta)[0] , ) )


            #[ coeff_X , _ , _  ] = L_SOLVER.l1ls( sP , b ,lmbda= 0.01 , tar_gap= 0.01);

            coeff_X, resid, grad, info = spg_bpdn( P, yeta, sigma)

            iClass_norm = [-1] * 6 ; ##np.zeros( (6 , 1) )

            for iClass in range(1 , 6):

                lastGuy = self.last_indices[iClass - 1];
                thisGuy = self.last_indices[iClass];

                # NEED -1 HERE
                A_iClass = self.A_iClass[iClass - 1]

                iClass_coeff = coeff_X[lastGuy + 1 : thisGuy  + 1];

                iVec = np.dot(A_iClass , iClass_coeff) ;

                iClass_norm[iClass] = np.linalg.norm(iVec)

            y_predict[glob_index] = np.argmax(iClass_norm)

        return y_predict