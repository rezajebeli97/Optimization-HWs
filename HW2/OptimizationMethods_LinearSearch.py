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
    [[tmp2_makhraj]] = np.matmul(y.T, s)
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
    steplengths = []
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        p_k = -1 * np.array(g)  # p_k : steepest direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        steplengths.append(alpha)
        x = x + alpha * p_k
        f_x = f.f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x, steplengths

def newton(f, maxIteration, maxAlpha):
    x = f.initialState()
    f_x = f.f(x)
    iteration = 0
    steplengths = []
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        g2 = f.hessian(x)
        p_k = np.matmul(np.linalg.inv(-1 * np.array(g2)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        steplengths.append(alpha)
        x = x + alpha * p_k
        f_x = f.f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x, steplengths

def quasi_newton_BFGS(f, maxIteration, maxAlpha):
    prevx = None
    x = f.initialState()
    f_prevx = None
    f_x = f.f(x)
    iteration = 0
    steplengths = []
    fx1_fx = []
    b = f.hessian(x)
    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = f.gradient(x)
        if iteration != 0:
            y = np.array(g) - np.array(f.gradient(prevx))
            s = x - prevx
            b = estimate_hessian(x, g, b, y, s)

        p_k = np.matmul(np.linalg.inv(-1 * np.array(b)), np.array(g))                 # p_k : newton direction
        alpha = stepLength_backtracking(f, maxAlpha, x, g, p_k, 0.5, 0.5)
        steplengths.append(alpha)
        prevx = x
        x = x + alpha * p_k
        f_prevx = f_x
        f_x = f.f(x)
        fx1_fx.append(abs(f_x - f_prevx))
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x, steplengths, fx1_fx


def draw(x):
    plotter.ylabel('Step Length')
    plotter.xlabel('step')
    plotter.plot(x)
    plotter.show()


f = Rosenbrock()
beta, steplengths, fx1_fx = quasi_newton_BFGS(f, 500, 1)
draw(fx1_fx)