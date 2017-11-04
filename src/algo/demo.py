import numpy as np

import matplotlib.pyplot as plt

# % Create random m-by-n encoding matrix and sparse vector
m = 50
n = 128
k = 14
[A, Rtmp] = np.linalg.qr(np.random.randn(n, m), 'reduced')
A = A.T
p = np.random.permutation(n)
p = p[0:k]
x0 = np.zeros(n)
x0[p] = np.random.randn(k)


# % -----------------------------------------------------------
# % Solve the basis pursuit denoise (BPDN) problem:
# %
# %    minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1
# %
# % -----------------------------------------------------------
from ext_lib.spgl1.spgl1 import spg_bpdn

print('%s ' % ('-' * 78))
print('Solve the basis pursuit denoise (BPDN) problem:      ')
print('                                              ')
print('  minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1')
print('                                              ')
print('%s%s ' % ('-' * 78, '\n'))



# % Set up vector b, and run solver
b = A.dot(x0) + np.random.randn(m) * 0.075

sigma = 0.10  # % Desired ||Ax - b||_2

x, resid, grad, info = spg_bpdn(A, b, sigma)


print(x)
print(resid)

plt.figure()
plt.plot(x, 'b')

plt.plot(x0, 'ro')
plt.legend(('Recovered coefficients', 'Original coefficients'))
plt.title('(b) Basis Pursuit Denoise')

print('%s%s%s' % ('-' * 35, ' Solution ', '-' * 35))
print('See figure 1(b)')
print('%s%s ' % ('-' * 78, '\n'))



plt.show()