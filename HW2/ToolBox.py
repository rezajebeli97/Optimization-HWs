import numpy as np
import cvxpy as cp

n = 400

A = []
for i in range(1, n + 1):
    tmp = []
    for j in range(1, n + 1):
        tmp.append(1 / (i + j - 1))
    A.append(tmp)
A = np.array(A)

b = []
for i in range(n):
    b.append([1])
b = np.array(b)

P = np.matmul(A.T, A)
q = -2 * np.dot(A.T, b)
r = np.dot(b.T, b)
x = cp.Variable(n)

# x^𝑇 P x + q^𝑇 x + r
objective = cp.Minimize(cp.quad_form(x, P) + q.T * x + r)
constraints = [-1000000 <= x, x <= 1000000]
prob = cp.Problem(objective, constraints)
print('Optimal objective value using cvxpy: ', prob.solve())