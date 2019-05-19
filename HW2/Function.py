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
    def f(self, x):
        n = np.shape(x)[0]
        b = []
        for i in range(n):
            b.append([1])

        A = []
        for i in range(1, n + 1):
            tmp = []
            for j in range(1, n + 1):
                tmp.append(1 / (i + j - 1))
            A.append(tmp)

        return np.linalg.norm(np.matmul(A, x) - b)

    def gradient(self, x):
        n = np.shape(x)[0]
        b = []
        for i in range(n):
            b.append([1])

        A = []
        for i in range(1, n + 1):
            tmp = []
            for j in range(1, n + 1):
                tmp.append(1 / (i + j - 1))
            A.append(tmp)

        b = np.array(b)
        A = np.array(A)
        return np.matmul(np.matmul(A.T, A), x) - 2 * np.matmul(A.T, b)

    def hessian(self, x):
        n = np.shape(x)[0]

        A = []
        for i in range(1, n + 1):
            tmp = []
            for j in range(1, n + 1):
                tmp.append(1 / (i + j - 1))
            A.append(tmp)

        A = np.array(A)
        return np.matmul(A.T, A)

    def initialState(self):
        x = []
        for i in range(2):
            x.append([0])
        return x