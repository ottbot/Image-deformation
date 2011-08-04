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

        self.dV = VectorFunctionSpace(self.mesh, 'DG', deg_cont - 1, dim=2)
        self.V  = VectorFunctionSpace(self.mesh, 'CG', deg_cont, dim=2)

        self.sigma_sq = 1
        self.alpha_sq = 1
        
        # Template
        self.qA_exp = Expression(('sin(x[0])','cos(x[0])'))
        self.qA = interpolate(self.qA_exp, self.V)

        # Target
        self.qB_exp = Expression(('2*sin(x[0])','3*cos(x[0])'))
        self.qB = interpolate(self.qB_exp, self.V)

        # initialize arrays
        self.U =  [Function(self.V) for i in xrange(self.N)]
        self.Q =  [Function(self.V) for i in xrange(self.N)] 
        self.Qh = [Function(self.V) for i in xrange(self.N)] 
        self.dS = [Function(self.V) for i in xrange(self.N)] 

        # set initial guess for U
        self.init_U()
        # calculate q at each timestep using initial guess
        self.calc_Q()


    def opt(self):
        return sp.optimize.fmin_bfgs(self.S, self.U, fprime=self.calc_dS)

        
    def init_U(self):
        u = Expression(('.1*sin(x[0])','.1*cos(x[0])'))

        U = np.zeros((np.shape(self.qA.vector().array())[0],self.N))

        for n in xrange(self.N):
            un = interpolate(u, self.V)
            noise = un.vector().array()
            self.U[n].assign(un)

 
    def calc_Q(self):
        q_prev = self.qA

        r = TestFunction(self.V)
        q = TrialFunction(self.V)

        a = dot(r,q)*dx
        A = assemble(a)        

        q = Function(self.V)   # the unknown at a new time level

        for n in xrange(self.N):
            L = dot(q_prev,r)*dx -  self.dt*dot(r,self.U[n])*dx
            b = assemble(L)

            solve(A, q.vector(), b)

            q_prev.assign(q)

            self.Q[n].assign(q)

            
    def qh_at_t1(self):
        # Find q hat at t = 1
        q1 = self.Q[-1]
        qB = self.qB 
        
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

        u = Function(self.V)
        q = Function(self.V)
        
        for n in reversed(xrange(self.N)):
            u.assign(self.U[n])
            q.assign(self.Q[n])
            j = self.j(q)

            c = .5*dot(u,u)/j - \
                (self.alpha_sq/2.0)*dot(u.dx(0),u.dx(0))/dot(q.dx(0),q.dx(0))

            L = dot(p,qh)*dx - c*dot(p.dx(0),qh.dx(0))*self.dt*dx
            
            b = assemble(L)

            solve(A, qh_prev.vector(), b)

            qh.assign(qh_prev)

            self.Qh[n].assign(qh)

                                          
    def j(self, q):
        dq = q.dx(0)
        return  sqrt(dot(dq,dq))
        

    def S(self, U):

        S = 0

        for n in xrange(self.N):
            j = self.j(self.Q[n])

            a = (dot(U[n],U[n])*j)*dx + .5*(self.alpha_sq*dot(U[n].dx(0), U[n].dx(0))/j)*dx
            #S += .5*assemble(energy_norm(a,U[n]))*self.dt
            S += assemble(a)

        S = 0.5*S*self.dt

        diff = self.Qh[-1] - self.qB
        err = assemble(inner(diff,diff)*dx)

        return S + 1/(self.sigma_sq)*err


        
    def calc_dS(self, U):

        v = TestFunction(self.V)
        dS = TrialFunction(self.V)
        
        a = dot(v,dS)*dx
        A = assemble(a)

        for n in xrange(self.N):
            u = U[n]
            j = self.j(self.Q[n])
            qh = self.Qh[n]

            L = dot(v,j*u)*dx + self.alpha_sq*dot(v.dx(0),u.dx(0))/j*dx - dot(v,qh)*dx
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

        for q in Q:
            X,Y = self.split_array(q.vector().array())

            line.set_data(X[sorted], Y[sorted])

            plt.draw()

    def plot_du(self, Q):
        dv = TestFunction(o.dV)
        du = TrialFunction(o.dV)
        
        q = Q

        a = dot(dv,du)*dx 
        l = dot(dv,q.dx(0))*dx

        A = assemble(a)
        b = assemble(l)

        du0 = Function(o.dV)
        solve(A, du0.vector(), b)


        #plt.figure()
        #plt.axis('equal')

        dX, dY = self.split_array(du0)
        print np.min(np.sqrt(dX**2+dY**2))


        #        plt.show()              


    def test(self, eps = 1e-10):
        v0 = TestFunction(self.V)

        dS0 = TrialFunction(self.V)

        a = dot(v0,dS0)*dx
        
        v = Expression(('cos(x[0])','cos(x[0])'))
        v = interpolate(v, self.V)

        i = 0

        for n in xrange(self.N):
            dS = self.dS[n]

            frm = action(action(a,v), dS)
            x = assemble(frm)

            i += x

        i *= self.dt

        print "sum: ", i
        
        

        # calculate the the value of the limit as eps at 1e-10 to 1e-20
        eps = np.array([10**(-n) for n in np.linspace(10.0,17,10)])
        lims = np.array([self.derivative_from_limit(v, x) for x in eps])


        #plt.figure()
        #plt.plot(lims/i)

        for l in lims:
            print "limit: ",l, 'solver: ', i

    def derivative_from_limit(self,v, eps = 1e-10):
        #S = self.S()

        #var = 0
        #for n in xrange(self.N):
        #    var += assemble(dot(self.dS[n], v)*dx)


        
        #lim (S[u + eps*v] - S[u])/eps
        # but: S[u +eps*v] = S[u] + eps*<dU/dS,v>
        # so just return <dU/dS, v>
        #return (eps*var)/eps

                           

        return (self.S(self.pert_u(v,eps)) - self.S())/eps
        

    # utility functions



    def pert_u(self, v, eps):
        Up =  [Function(self.V) for i in xrange(self.N)] 

        for n in xrange(self.N):
          Up[n].assign(self.np_to_coeff(self.U[n].vector().array() + eps*v.vector().array()))


        return Up

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


