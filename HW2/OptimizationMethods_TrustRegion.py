import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter
from Function import *


# np.seterr(divide='ignore', invalid='ignore')

def cauchy_point(g, b, delta):
    ps = ((-1 * delta) / np.linalg.norm(g)) * g

    [[condition]] = np.matmul(np.matmul(g.T, b), g)
    if condition <= 0:
        tou = 1
    else:
        tou = min(1, np.linalg.norm(g) ** 3 / (delta * condition))
    pc = tou * ps  # cauchy point
    return pc


def dogleg(g, B, delta):
    pb = np.matmul(- 1 * np.linalg.inv(B), g)
    pu = - (np.matmul(g.T, g) / np.matmul(np.matmul(g.T, B), g)) * g
    tou = delta / np.linalg.norm(pu)
    if tou >= 0 and tou <= 1:
        p = tou * pu
    elif tou < 2:
        p = pu + (tou - 1) * (pb - pu)
    else:
        pb_norm = np.linalg.norm(pb)
        if pb_norm <= delta:
            p = pb
        else:
            p = (delta / pb_norm) * pb

    return p


def trust_region(f, delta0, efficent, maxIteration,
                 method):  # delta0 bigger than 0 , first delta should initialize sth between 0 and delta0, efficient should be sth between 0 and 0.25
    delta = delta0 / 2
    x = f.initialState()
    f_x = f.f(x)
    iteration = 0
    efficencies = []
    fx1_fx = []

    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = np.array(f.gradient(x))
        b = np.array(f.hessian(x))

        if method == "cauchy":
            p = cauchy_point(g, b, delta)
        elif method == "dogleg":
            p = dogleg(g, b, delta)

        m_0 = f.f(x)
        m_p = f.f(x) + np.matmul(p.T, g) + 0.5 * np.matmul(np.matmul(p.T, b), p)  # [[]]
        [[efficency]] = (f.f(x) - f.f(x + p)) / (m_0 - m_p)
        efficencies.append(efficency)

        if efficency < 0.25:
            delta = 0.25 * delta
        else:
            if efficency > 0.75 and abs(np.linalg.norm(p) - delta) < 0.000001:
                delta = min(2 * delta, delta0)
            else:
                delta = delta

        if efficency > efficent:
            x = x + p
        else:
            x = x

        f_x = f.f(x)

        fx1_fx.append(abs(f_x - m_0))

        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x, efficencies, fx1_fx


def draw(x):
    plotter.ylabel('|f(xk+1) - f(xk)|')
    plotter.xlabel('step')
    plotter.plot(x)
    plotter.show()


f = Rosenbrock()
x_star, efficencies, fx1_fx = trust_region(f, 1.5, 0.15, 10000, "dogleg")
draw(fx1_fx)
