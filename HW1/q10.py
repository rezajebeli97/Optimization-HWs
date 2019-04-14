import cvxpy as cp

x1 = cp.Variable(1)
x2 = cp.Variable(1)

constraints = [2*x1+x2 >= 1, x1+3*x2 >= 1, x1 >= 0, x2 >= 0]

objective_functions = [
    cp.Minimize(x1 + x2),
    cp.Minimize(-x1 - x2),
    cp.Minimize(x1),
    cp.Minimize(cp.maximum(x1, x2)),
    cp.Minimize(cp.power(x1, 2) + 9*cp.power(x2, 2))
]

for f_id, f in enumerate(objective_functions):
    problem = cp.Problem(f, constraints)
    print('function', f_id, ':')
    print('Optimal function value    :', problem.solve())
    print('Optimal variables: x1={}, x2={}'.format(x1.value, x2.value))
    print('\n')