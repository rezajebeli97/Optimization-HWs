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


    def f(self, x):                       # 100(x2 - x1^2)^2 + (1 - x1)^2
        [x1] = x[0]
        [x2] = x[1]
        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


    def gradient(self, x):                # [     200(x2 - x1^2)*(-2x1) + 2(1-x1)(-x1)      ,       200(x2-x1^2)    ]
        [x1] = x[0]
        [x2] = x[1]
        g1 = 200 * (x2 - x1 ** 2) * (-2 * x1) + 2 * (x1 - 1)
        g2 = 200 * (x2 - x1 ** 2)
        g = [[g1], [g2]]
        return g


    def hessian(self, x):
        [x1] = x[0]
        [x2] = x[1]
        g11 = -400 * x2 + 1200 * x1 ** 2 + 2
        g12 = -400 * x1
        g21 = -400 * x1
        g22 = 200
        g = [[g11, g12], [g21, g22]]
        return g


    def initialState(self):
        return [[-1.2], [1]]


class LeastSquare(Function):
    def __init__(self, n):
        self.A = []
        self.n = n
        for i in range(1, n + 1):
            tmp = []
            for j in range(1, n + 1):
                tmp.append(1 / (i + j - 1))
            self.A.append(tmp)

        self.b = []
        for i in range(n):
            self.b.append([1])

    def f(self, x):
        return np.linalg.norm(np.matmul(self.A, x) - self.b) ** 2

    def gradient(self, x):
        self.b = np.array(self.b)
        self.A = np.array(self.A)
        return 2 * np.matmul(np.matmul(self.A.T, self.A), x) - 2 * np.matmul(self.A.T, self.b)

    def hessian(self, x):
        self.A = np.array(self.A)
        return 2 * np.matmul(self.A.T, self.A)

    def initialState(self):
        x = []
        for i in range(self.n):
            x.append([1])
        return x