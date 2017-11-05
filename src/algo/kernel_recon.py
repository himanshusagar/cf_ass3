import gc
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

# import l1ls as L_SOLVER

POWER = 2;


def PolyKernel(x , y , gamma=None):

    if gamma is None:
        gamma = 1.0 / x.shape[1]
#    gamma = 1.0;

    #return 1 + np.dot( x.T , y );
    #return  np.square(1 +  np.multiply(gamma , np.dot( x.T , y )) );
    return  np.power(1 + gamma*np.dot( x.T , y ) , POWER);


from sklearn.metrics.pairwise import polynomial_kernel


class KernelRecon:
    def __init__(self):
        self.data_clasees = [-1, 1, 2, 3, 4, 5];
        self.last_indices = [-1, -1, -1, -1, -1, -1];
        self.B_t_B_iClass = []
        self.scalar = StandardScaler();

    def fit(self, X, y):
        #
        # self.X = X;
        # self.Y = y;

        # self.scalar.fit(X)
        # X = self.scalar.transform(X)


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
        self.internal_A =  dataXY[: self.n - 1, :]

        B_t_B = PolyKernel(self.internal_A , self.internal_A)
       # asdas = polynomial_kernel( self.internal_A ,self.internal_A , degree=2 , coef0=1 , gamma=1)


        self.pca = PCA(n_components=40);
        self.pca.fit( np.transpose(B_t_B) );
        print("Variance")
        print(np.sum(self.pca.explained_variance_ratio_))

        if hasattr(self, 'pca'):
            self.B_t_B = np.transpose(self.pca.transform( np.transpose(B_t_B) ) );
        else:
            self.B_t_B = B_t_B;

        print("self.last indices")
        print(self.last_indices)

        for iClass in range(1, 6):
            lastGuy = self.last_indices[iClass - 1];
            thisGuy = self.last_indices[iClass];
            self.B_t_B_iClass.append( self.B_t_B[:, lastGuy + 1: thisGuy + 1] )


    def predict(self, X_test):

        # X_test = self.scalar.transform(X_test)

        #X_test = preprocessing.normalize(X_test , norm='l2', axis=0)

        y_predict = np.zeros((np.shape(X_test)[0] , 1))


        sigma = 0.01  # % Desired ||Ax - b||_2

        print('%s ' % ('-' * 78))
        print('KernelRecon the basis pursuit denoise (BPDN) problem:      ')
        print('                                              ')
        print('  minimize ||x||_1 subject to ||Ax - b||_2 <= ' , sigma)
        print('                                              ')
        print('%s%s ' % ('-' * 78, '\n'))

        LIMIT = len(y_predict);

        print()
        gc.collect();

        for glob_index in range( LIMIT ):
            print("#" + str(glob_index) , " ")

            # % Set up vector b, and run solver
            internal_b = X_test[glob_index];
            #
            # print("b_shape" , np.shape(b));

            b_t_phi = PolyKernel(self.internal_A , internal_b )

            if hasattr(self , 'pca'):
                ### Apply Transform
                b_t_phi = b_t_phi.reshape( -1 , 1)
                b_t_phi = np.transpose(self.pca.transform( np.transpose(b_t_phi ) ) )
                b_t_phi = np.reshape(b_t_phi , (np.shape(b_t_phi)[0] , ))




            coeff_X, resid, grad, info = spg_bpdn(self.B_t_B , b_t_phi, sigma)

            from scipy import sparse

            #[ coeff_X , _ , _  ] = L_SOLVER.l1ls( sparse.csr_matrix(A) , b ,lmbda= 0.01 , tar_gap= 0.01);
            ##coeff_X = np.linalg.lstsq(A , b)[0]

            #print("coeff x" , np.shape(coeff_X))

            iClass_norm = [-1] * 6 ; ##np.zeros( (6 , 1) )

            for iClass in range(1 , 6):

                lastGuy = self.last_indices[iClass - 1];
                thisGuy = self.last_indices[iClass];

                # NEED -1 HERE
                B_t_B_iClass = self.B_t_B_iClass[iClass - 1]

                iClass_coeff = coeff_X[lastGuy + 1 : thisGuy  + 1];

                iVec = np.dot(B_t_B_iClass , iClass_coeff);

                iClass_norm[iClass] = np.linalg.norm(iVec)

            y_predict[glob_index] = np.argmax(iClass_norm)

        return y_predict