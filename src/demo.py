from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np

X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
y = [0, 1, 0, 1]

K = polynomial_kernel(X=X ,  degree=2 , coef0=1 , gamma=1)

print(np.shape(X))

print(K)
print(np.shape(K))

glob_pca = PCA(n_components=2);

glob_pca.fit( K );

new_K = glob_pca.transform(K)
new_y = glob_pca.transform(y)

print("Variance")
print(np.sum(pca.explained_variance_ratio_))


svm = SVC(kernel='precomputed').fit(K, y)

print(svm.predict(K))
