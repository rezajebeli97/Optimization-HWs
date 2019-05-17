import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter


def f(x):  # 100(x2 - x1^2)^2 + (1 - x1)^2
    [x1] = x[0]
    [x2] = x[1]
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def gradient(x):  # [     200(x2 - x1^2)*(-2x1) + 2(1-x1)(-x1)      ,       200(x2-x1^2)    ]
    [x1] = x[0]
    [x2] = x[1]
    g1 = 200 * (x2 - x1 ** 2) * (-2 * x1) + 2 * (x1 - 1)
    g2 = 200 * (x2 - x1 ** 2)
    g = [[g1], [g2]]
    return g

def hessian(x):
    [x1] = x[0]
    [x2] = x[1]
    g11 = -400 * x2 + 1200 * x1**2 + 2
    g12 = -400 * x1
    g21 = -400 * x1
    g22 = 200
    g = [[g11, g12], [g21 , g22]]
    print(g)
    return g

def estimate_hessian(x, gradient, b, y, s):

    return g


def initialState():
    return [[-1.2], [1]]


def sufficient_decrease_condition_wolfie1(alpha, x_k, gradient_k, p_k, c):
    f_nextx = f(x_k + alpha * p_k)
    [[supposed_f]] = f(x_k) + c * alpha * np.matmul(np.array(gradient_k).T, p_k)
    if f_nextx <= supposed_f:
        return True
    else:
        return False


def stepLength_backtracking(maxAlpha, x_k, gradient_k, p_k, ro, c):  # ro:zaribe kaheshe alpha     ,   c: mizane sakht girie wolfe1 condition
    alpha = maxAlpha
    while True:
        if sufficient_decrease_condition_wolfie1(alpha, x_k, gradient_k, p_k, c):
            return alpha
        else:
            alpha = ro * alpha


def gradient_descent(maxIteration, maxAlpha):
    x = initialState()
    f_x = f(x)
    iteration = 0
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = gradient(x)
        p_k = -1 * np.array(g)  # p_k : steepest direction
        alpha = stepLength_backtracking(maxAlpha, x, g, p_k, 0.5, 0.5)
        print(f_x)
        x = x + alpha * p_k
        f_x = f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x

def newton(maxIteration, maxAlpha):
    x = initialState()
    f_x = f(x)
    iteration = 0
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = gradient(x)
        g2 = hessian(x)
        p_k = np.matmul(np.linalg.inv(-1 * np.array(g2)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(maxAlpha, x, g, p_k, 0.5, 0.5)
        print(x)
        print(f_x)
        print()
        x = x + alpha * p_k
        f_x = f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x

def quasi_newton_BFGS(maxIteration, maxAlpha):
    x = initialState()
    f_x = f(x)
    iteration = 0
    b = hessian(x)
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = gradient(x)
        b = estimate_hessian(x, gradient, b, y, s)
        p_k = np.matmul(np.linalg.inv(-1 * np.array(b)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(maxAlpha, x, g, p_k, 0.5, 0.5)
        print(x)
        print(f_x)
        print()
        x = x + alpha * p_k
        f_x = f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x

def compare(beta):
    a = np.load('data.npz')
    x_test = []
    for i in range(2000):
        x_test.append([1, a['x1_test'][i], a['x2_test'][i] ** 2, a['x2_test'][i] ** 2 * a['x1_test'][i]])
    x1_test = [[i] for i in a['x1_test']]
    x2_test = [[i] for i in a['x2_test']]
    y_test = [[i] for i in a['y_test']]
    y_predict = np.matmul(x_test, beta)
    sse_test = np.linalg.norm(np.subtract(y_test, y_predict), 2) ** 2
    sse_train = f(beta)
    print("SSE for train data : ", sse_train)
    print("SSE for test data : ", sse_test)
    fig = plotter.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1_test, x2_test, y_test, color='red')
    ax.scatter(x1_test, x2_test, y_predict, color='blue')
    plotter.show()


beta = newton(10000, 1)
# beta = stochastic(5000, 0.01)
# compare(beta=beta)