import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter
from math import pi, cos, sin
from random import random
from Function import *

def model(p):
    p = np.array(p)
    x = f.initialState()
    g = f.gradient(x)
    b = f.hessian(x)
    return f.f(x) + np.matmul(p.T, g) + 0.5 * np.matmul(np.matmul(p.T, b), p)  # [[]]

def contour(point_nums=10000):
    points = []
    for i in range(int((max_m + 1) / 10)):
        points.append([[], []])
    for i in range(point_nums):
        rand1 = random()
        rand2 = random()
        length = rand1 * 2
        theta = rand2 * 2 * pi
        px = cos(theta) * length
        py = sin(theta) * length
        m = model([[px], [py]])

        if m <= max_m:
            [x0] = f.initialState()[0]
            [x1] = f.initialState()[1]
            points[int(m / 10)][0].append(px + x0)
            points[int(m / 10)][1].append(py + x1)

    _, ax = plotter.subplots()
    for m_xs in points:
        if len(m_xs[0]) > 0:
            mean_x = m_xs[0]
            mean_y = m_xs[1]
            plotter.scatter(mean_x, mean_y, marker='.', alpha=.5)
    plotter.show()


def dogleg(g, B, delta):
    g = np.array(g)
    B = np.array(B)
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


f = Rosenbrock()
min_m, max_m = 0, 119
# contour()

if __name__ == '__main__':
    contour()
    _, ax = plotter.subplots()
    for delta in [.1, .4, .8, 2]:
        p = dogleg(f.gradient(f.initialState()), f.hessian(f.initialState()), delta)
        plotter.scatter(p[0], p[1], marker='x', c='purple')
        ax.annotate(delta, (p[0], p[1]))

    plotter.show()








