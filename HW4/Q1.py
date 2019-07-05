import time
import numpy as np
from Function import Quadratic


def multiply(A, B, C):
    return np.matmul(np.matmul(A, B), C)

t0 = int(round(time.time() * 1000))

n = 100
add_noise = True
tridiag_or_hilb = True

# tridiag
if tridiag_or_hilb:
    tmp = -1 * np.ones([n-1])
    diag = 4 * np.ones([n])
    A = np.diag(tmp, -1) + np.diag(diag, 0) + np.diag(tmp, 1)

# hilb
else:
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + j + 1)
    A = np.array(A)

# add noise
if add_noise:
    e = 0.000001
    A = A + e

b = []
for i in range(n):
    b.append([1])

func = Quadratic(A, b, n)

x = func.initialState()
r = np.matmul(func.A, x) - func.b
p = -r
k = 0

while np.linalg.norm(r, 2) > 0.0001:
    alpha = - np.matmul(r.T, p) / multiply(p.T, func.A, p)
    x = x + alpha * p
    r = np.matmul(func.A, x) - func.b
    beta = multiply(r.T, func.A, p) / multiply(p.T, func.A, p)
    p = -r + beta * p
    k = k + 1

t = int(round(time.time() * 1000))

print("x* = ", x)
print("f(x*) = ", func.f(x))
print("time(ms) = ", t - t0)
print("||Ax* - b|| = ", np.linalg.norm(np.matmul(func.A, x) - func.b, 2))