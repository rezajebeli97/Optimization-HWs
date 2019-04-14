import cvxpy as cp

x1 = cp.Variable(1)
x2 = cp.Variable(1)

objective_function = cp.Minimize(x1 + 3*x2)
constraints = [-x1+x2 <= 2, x1+x2 >= 2, x2 >= 0, 2*x1-3*x2 <= 5]
problem = cp.Problem(objective_function, constraints)

print('Optimal function value    :', problem.solve())
print('optimal variables: x1 = {}, x2 = {}'.format(x1.value, x2.value))