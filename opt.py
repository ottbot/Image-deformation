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

        self.sigma_sq = 1
        self.alpha_sq = 1
        
        # Template
        self.qA = Expression(('sin(x[0])','cos(x[0])'))
        # Target
        self.qB = Expression(('2*sin(x[0])','2*cos(x[0])'))


        self.U =  [Function(self.V) for i in xrange(self.N)]
        self.Q =  [Function(self.V) for i in xrange(self.N)] 
        self.Qh = [Function(self.V) for i in xrange(self.N)] 
        self.dS = [Function(self.V) for i in xrange(self.N)] 

        
    def set_U(self,U):
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

        r = TestFunction(self.V)
        q = TrialFunction(self.V)

        a = dot(r,q)*dx
        A = assemble(a)        


        q = Function(self.V)   # the unknown at a new time level

        t = 0.0

        for n in xrange(self.N):
            L = dot(q_prev,r)*dx -  self.dt*dot(r,self.U[n])*dx
            b = assemble(L)

            solve(A, q.vector(), b)

            t += self.dt
            q_prev.assign(q)

            self.Q[n] = q

    def qh_at_t1(self):
        # Find q hat at t = 1
        q1 = self.Q[-1]
        qB = interpolate(self.qB, self.V)
        
        p = TestFunction(self.V)
        qh1 = TrialFunction(self.V)

        a = dot(p,qh1)*dx
        L = -1.0/self.sigma_sq* dot(p,q1 - qB)*dx

        A = assemble(a)
        b = assemble(L)

        qh1 = Function(self.V)
        solve(A,qh1.vector(),b)

        return qh1
    
    def qh(self):
        qh = self.qh_at_t1() 

        # Find q hat at each time step by stepping backwards in time from qh1
        p = TestFunction(self.V)
        qh_prev = TrialFunction(self.V)

        
        a = dot(p, qh_prev)*dx
        A = assemble(a)

        qh_prev = Function(self.V) # unknown at next timestep
        
        t = 1.0

        for n in reversed(xrange(self.N)):
            u = self.U[n]
            q = self.Q[n]
            j = self.j(q)

            c = .5*dot(u,u)/j - \
                (self.alpha_sq/2.0)*dot(u.dx(0),u.dx(0))/dot(qh.dx(0),qh.dx(0))

            L = dot(p,qh_prev)*dx - c*dot(p.dx(0),qh.dx(0))*self.dt*dx
            
            b = assemble(L)

            solve(A, qh_prev.vector(), b)

            qh.assign(qh_prev)

            self.Qh[n] = qh
            print t
            t -= self.dt
                                          
    def j(self, q):
        dq = q.dx(0)
        return  sqrt(dot(dq,dq))
        

    def S(self):
        S = 0
        n = 0

        v = TestFunction(self.V)
        u = TrialFunction(self.V)
        

        for n in xrange(self.N):
            j = self.j(self.Q[n])
            a = dot(v,u)*j*dx + (self.alpha_sq*dot(v.dx(0), u.dx(0))/j)*dx
            S += .5*assemble(energy_norm(a, self.U[n]))*self.dt



        qB = interpolate(self.qB, self.V)
        err = self.np_to_coeff(self.Q[-1].vector().array() - qB.vector().array())

        a = dot(v,u)*dx

        l2norm = sqrt(assemble(energy_norm(a,err)))
        return S + 1/(2.0*self.sigma_sq)*l2norm


        
    def calc_dS(self):
        v = TestFunction(self.V)
        dS = TrialFunction(self.V)
        
        a = dot(v,dS)*dx
        A = assemble(a)

        for n in xrange(self.N):
            u = self.U[n]
            j = self.j(self.Q[n])
            qh = self.Qh[n]

            L = dot(v,j*u)*dx + self.alpha_sq*dot(v.dx(0),u.dx(0))/j*dx - dot(v,qh)*ds
            
            b = assemble(L)

            solve(A, self.dS[n].vector(), b)
            

    def plot_steps(self,Q):

        plt.ion()
        plt.figure()
        plt.axis('equal')

        line, = plt.plot(*self.split_array(Q[0]))
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        
        sorted = self.get_sort_order(self.V)

        #for i in xrange(np.shape(self.Q)[1]):
        for q in Q:
            print "plottus Q"
            X,Y = self.split_array(q.vector().array())
            line.set_data(X[sorted], Y[sorted])

            plt.draw()


    # utilitiy (private?) functions
    def np_to_coeff(self,arr):
        f = Function(self.V)
        f.vector()[:] = arr
        return f

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

o.qh()
