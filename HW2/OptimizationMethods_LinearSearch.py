import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter
from Function import *

def estimate_hessian(x, gradient, b, y, s):
    tmp1 = np.matmul(np.matmul(np.matmul(b,s), s.T), b)
    [[tmp1_makhraj]] = np.matmul(np.matmul(s.T, b), s)
    tmp1 = tmp1 / tmp1_makhraj

    tmp2 = np.matmul(y, y.T)
    [[tmp2_makhraj]] = np.matmul(y.T , s)
    tmp2 = tmp2 / tmp2_makhraj

    return b - tmp1 + tmp2


def sufficient_decrease_condition_wolfie1(f, alpha, x_k, gradient_k, p_k, c):
    f_nextx = f.f(x_k + alpha * p_k)
    [[supposed_f]] = f.f(x_k) + c * alpha * np.matmul(np.array(gradient_k).T, p_k)
    if f_nextx <= supposed_f:
        return True
    else:
        return False


def stepLength_backtracking(f, maxAlpha, x_k, gradient_k, p_k, ro, c):  # ro:zaribe kaheshe alpha     ,   c: mizane sakht girie wolfe1 condition
    alpha = maxAlpha
    while True:
        if sufficient_decrease_condition_wolfie1(f, alpha, x_k, gradient_k, p_k, c):
            return alpha
        else:
            alpha = ro * alpha


def gradient_descent(f, maxIteration, maxAlpha):
    x = f.initialState()
    f_x = f.f(x)
    iteration = 0
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        p_k = -1 * np.array(g)  # p_k : steepest direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        x = x + alpha * p_k
        f_x = f.f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x

def newton(f, maxIteration, maxAlpha):
    x = f.initialState()
    f_x = f.f(x)
    iteration = 0
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        g2 = f.hessian(x)
        p_k = np.matmul(np.linalg.inv(-1 * np.array(g2)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        x = x + alpha * p_k
        f_x = f.f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x

def quasi_newton_BFGS(f, maxIteration, maxAlpha):
    prevx = None
    x = f.initialState()
    f_x = f.f(x)
    iteration = 0
    b = f.hessian(x)
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        if iteration != 0:
            y = np.array(g) - np.array(f.gradient(prevx))
            s = x - prevx
            b = estimate_hessian(x, g, b, y, s)

        p_k = np.matmul(np.linalg.inv(-1 * np.array(b)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        prevx = x
        x = x + alpha * p_k
        f_x = f.f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x


f = LeastSquare()
beta = gradient_descent(f, 10000, 1)
# beta = stochastic(5000, 0.01)
# compare(beta=beta)