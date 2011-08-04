import scipy as sp
from scipy.optimize import fmin_bfgs

class OptEx:

    x0 = 2

    def f(self, x):
        return x**4 + 2*x**3 + x**2 + 3*x

    def fprime(self, x):
        return 4*x**3 + 6*x**2 + 2*x + 3


    def opt(self):
        return fmin_bfgs(self.f, self.x0, fprime=self.fprime)
