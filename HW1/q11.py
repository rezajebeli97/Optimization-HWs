import cvxpy as cp
import numpy as np

P = np.array([
    [13, 12, -2],
    [12, 17, 6],
    [-2, 6, 12]
])
q = np.array([-22, -14.5, 13])
r = 1

x = cp.Variable(3)
objective_function = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T * x + r)
constraints = [-1 <= x, x <= 1]
problem = cp.Problem(objective_function, constraints)
print('Optimal function value    :  ', problem.solve())
print('Optimal variables         :   x = {}'.format(x.value))