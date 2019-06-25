from abc import ABCMeta, abstractmethod
import numpy as np

class Function:
    __metaclass__ = ABCMeta

    @classmethod
    def version(self): return "1.0"

    @abstractmethod
    def f(self): raise NotImplementedError

    @abstractmethod
    def gradient(self): raise NotImplementedError

    @abstractmethod
    def hessian(self): raise NotImplementedError

    @abstractmethod
    def initialState(self): raise NotImplementedError

############################################################################

class Rosenbrock(Function):


    def f(self, x):                       # 10(x2 - x1^2)^2 + (1 - x1)^2
        [x1] = x[0]
        [x2] = x[1]
        return 10 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


    def gradient(self, x):                # [     200(x2 - x1^2)*(-2x1) + 2(1-x1)(-x1)      ,       200(x2-x1^2)    ]
        [x1] = x[0]
        [x2] = x[1]
        g1 = 20 * (x2 - x1 ** 2) * (-2 * x1) + 2 * (x1 - 1)
        g2 = 20 * (x2 - x1 ** 2)
        g = [[g1], [g2]]
        return g


    def hessian(self, x):
        [x1] = x[0]
        [x2] = x[1]
        g11 = -40 * x2 + 120 * x1 ** 2 + 2
        g12 = -40 * x1
        g21 = -40 * x1
        g22 = 20
        g = [[g11, g12], [g21, g22]]
        return g


    def initialState(self):
        return [[0], [-1]]