# todo: put dolfin in own namespace?
from dolfin import *
import numpy as np
import scipy as sp
import matplotlib.pylab as plt

class CurveOpt:

    def __init__(self, m = 100,n=100, deg_cont = 2, deg_dis = 1):
        self.M = m # number of nodal val
        self.N = n # number of timesteps

        self.dt = 1.0/n

        self.mesh = Interval(self.M, 0, 2*pi)


        #self.VD  = VectorFunctionSpace(self.mesh, 'DG', deg_dis, dim=2)
        #self.dVD = VectorFunctionSpace(self.mesh, 'DG', deg_dis - 1, dim=2)
        #self.VC  = VectorFunctionSpace(self.mesh, 'CG', deg_cont, dim=2)

        self.dV  = VectorFunctionSpace(self.mesh, 'CG', deg_cont - 1, dim=2)
        self.V  = VectorFunctionSpace(self.mesh, 'CG', deg_cont, dim=2)

        
        # Template
        self.qA = Expression(('sin(x[0])','cos(x[0])'))
        # Target
        self.qB = Expression(('2*sin(x[0])','2*cos(x[0])'))


        
        # TODO, get needed dims properly!
        #self.Q = np.zeros((4*self.M + 2,self.N))
        #self.U = [TrialFunction(self.VD) for i in xrange(self.N)]

        #self.U = [Expression(('sin(x[0])','cos(x[0])')) for i in xrange(self.N)]
        self.U = [Function(self.V) for i in xrange(self.N)]
        
        
    def set_U(self,U):
        # todo: raise error if shape mismatch
        for n in xrange(self.N):
            self.U[n].vector()[:] = U[:,n]

        
    def dummy_U(self):
        q = interpolate(self.qB, self.V)
        q = q.vector().array()
        U = np.zeros((np.shape(q)[0],self.N))

        for n in xrange(self.N):
            U[:,n] = np.random.rand(1)[0]*q

        return U
        
     
 
    def calc_Q(self):
        q_prev = interpolate(self.qA, self.V)

        if not hasattr(self, 'Q'):
            self.Q = np.zeros((np.shape(q_prev.vector().array())[0], self.N))

        r = TestFunction(self.V)
        q = TrialFunction(self.V)

        a = dot(r,q)*dx
        
        n = 0

        L = dot(q_prev,r)*dx -  self.dt*dot(r,self.U[n])*dx

        A = assemble(a)

        q = Function(self.V)   # the unknown at a new time level

        t = 0.0

        for n in xrange(self.N):
            b = assemble(L)

            solve(A, q.vector(), b)

            t += self.dt
            q_prev.assign(q)
            self.Q[:,n] = q.vector().array()
            print 'time ', t




    def S(self):
        S = 0
        n = 0

        v = TestFunction(self.V)
        u = TrialFunction(self.V)
        
        alpha_sq = 1

        #a = dot(v,self.U[n])*dx #+ (alpha_sq*dot(v.dx(0), self.U[n].dx(0)))*dx
        a = dot(v,u)*dx + (alpha_sq*dot(v.dx(0), u.dx(0)))*dx

        #m = dot(v,self.U[n])*dx        
        #l = alpha_sq*dot(v.dx(0), self.U[n].dx(0))*dx
        for n in xrange(self.N):
            # todo: sum
            S += .5*assemble(energy_norm(a, self.U[n]))*self.dt

            

        return S

    def plot_Q(self):

        plt.ion()
        plt.figure()
        plt.axis('equal')

        line, = plt.plot(*self.split_array(self.Q[:,0]))
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        
        sorted = self.get_sort_order(self.V)

        for i in xrange(np.shape(self.Q)[1]):
            X,Y = self.split_array(self.Q[:,i])
            line.set_data(X[sorted], Y[sorted])

            plt.draw()


    # utilitiy (private?) functions
    def split_array(self,q):
        if isinstance(q, np.ndarray):
            x = q
        else:
            x = q.vector().array()

        X = x[0:np.size(x)/2]
        Y = x[np.size(x)/2: np.size(x)]

        return X,Y

    def get_sort_order(self, fun_space):
        vals = interpolate(Expression(('x[0]','x[0]')), fun_space)
        return np.argsort(self.split_array(vals)[0])


o = CurveOpt(100,100)
U = o.dummy_U()
o.set_U(U)

o.calc_Q()

S = o.S()
