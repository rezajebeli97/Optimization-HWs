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
    return g


def initialState():
    return [[-1.2], [1]]


def cauchy_point(g, b, delta):
    ps = ((-1 * delta) / np.linalg.norm(g)) * g

    [[condition]] = np.matmul(np.matmul(g.T, b), g)
    if condition <= 0:
        tou = 1
    else:
        tou = min(1, np.linalg.norm(g) ** 3 / (delta * condition))
    pc = tou * ps  # cauchy point

    return pc


def trust_region(delta0, efficent, maxIteration, method):        #delta0 bigger than 0 , first delta should initialize sth between 0 and delta0, efficient should be sth between 0 and 0.25
    delta = delta0/2
    x = initialState()
    f_x = f(x)
    iteration = 0
    efficencies = []

    while f_x > 0.00000000000000000000000001 and iteration < maxIteration:
        g = np.array(gradient(x))
        b = np.array(hessian(x))

        if method == "cauchy":
            p = cauchy_point(g, b, delta)
        elif method == "dogleg":
            p = dogleg()

        m_0 = f(x)
        m_p = f(x + p) + np.matmul(p.T, g) + 0.5 * np.matmul(np.matmul(p.T, b), p) #[[]]
        [[efficency]] = (f(x) - f(x + p)) / (m_0 - m_p)
        efficencies.append(efficency)

        if efficency < 0.25:
            delta = 0.25 * delta
        else:
            if efficency > 0.75 and np.linalg.norm(p) == delta:
                delta = min(2 * delta, delta0)
            else:
                delta = delta

        if efficency > efficent:
            x = x + p
        else:
            x = x

        f_x = f(x)
        iteration += 1
    print("x = ", x)
    print("f(x) = ", f_x)
    return x , efficencies


def draw(x):
    plotter.ylabel('Ro')
    plotter.xlabel('step')
    plotter.plot(x)
    plotter.show()


x_star , efficencies = trust_region(1, 0.15, 1000, "cauchy")
# beta = stochastic(5000, 0.01)
draw(efficencies)