import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD




def get_U_DiagS_V(U, s, Vh):
    return np.dot(U , np.dot( np.diag(s) , Vh ) )


def apply_svd_get_nTh_approx(data, k_component):
    U , s , Vh = np.linalg.svd(data , full_matrices=False)

    assert np.allclose(data , get_U_DiagS_V(U , s , Vh) )

    print("Apply SVD : K = " + str(k_component) + " N = " + str(np.size(s)))
    s[k_component :] = 0
    kTh_approx = get_U_DiagS_V(U , s ,Vh)
    iScore_matrix = np.dot(U , np.diag(s))
    iLoad_matrix = np.transpose(Vh)

    return kTh_approx , iScore_matrix , iLoad_matrix

