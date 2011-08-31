from dolfin import *
import time as pytime
import numpy as np
import scipy as sp
#from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pylab as plt



class Immersion:

    # flag to test if the arrays of U,Q, Qh need to be populated
    # call populate_arrays if so 
    populated = False


    def __init__(self, m = 100,n=10, qA=None, qB=None,  deg_cont = 2, deg_dis = 1):
        self.M = m # number of nodal val
        self.N = n # number of timesteps

        self.dt = 1./n

        self.mesh = Interval(self.M, 0, 2*pi)

        self.dV = VectorFunctionSpace(self.mesh, 'DG', deg_cont - 1, dim=2)
        self.V  = VectorFunctionSpace(self.mesh, 'CG', deg_cont, dim=2)

        self.sigma_sq = 0.00001
        self.alpha_sq = 0.0001


        # TODO.. if tmpl/target is a string (tuple) take it as an expression
        #        if it's an array, do it 

        # Template
        if qA is None:        
            self.qA_exp = Expression(('sin(x[0] + 1)','cos(x[0] + 2)'))
            #self.qA_exp = Expression(('sin(x[0])','cos(x[0])'))
            self.qA = interpolate(self.qA_exp, self.V)
        else:
            self.qA = Function(self.V)
            self.qA.vector()[:] = qA


        if qB is None:
            self.qB_exp = Expression(('sin(x[0] - 1)','cos(x[0] - 2)'))
            #self.qB_exp = Expression(('1.5*sin(x[0])','0.8*cos(x[0])'))
            self.qB = interpolate(self.qB_exp, self.V)
        else:
            self.qB = Function(self.V)
            self.qB.vector()[:] = qB
            

        
        x, y = self.mat_shape = (np.shape(self.qB.vector().array())[0], self.N)
        self.template_size = x
        self.vec_size = x * y

        # initialize arrays
        self.U =  [Function(self.V) for i in xrange(self.N)]
        self.Q =  [Function(self.V) for i in xrange(self.N)] 
        self.Qh = [Function(self.V) for i in xrange(self.N)] 
        self.dS = [Function(self.V) for i in xrange(self.N)] 


 
    def calc_Q(self):
        r = TestFunction(self.V)
        q_next = TrialFunction(self.V)

        a = inner(r,q_next)*dx
        A = assemble(a)        

        q_next = Function(self.V)   # the unknown at a new time level
        q = Function(self.V)

        #initial Q

        q.assign(self.qA)

        for n in xrange(self.N):
            L = inner(q, r)*dx -  self.dt*inner(r,self.U[n])*dx
            b = assemble(L)

            solve(A, q_next.vector(), b)

            q.assign(q_next)

            self.Q[n].assign(q)


    def j(self, q):
        return  sqrt(inner(q.dx(0),q.dx(0)))
        # V = FunctionSpace(self.mesh, 'CG', 1)
        # j = inner(q.dx(0),q.dx(0))
        # pj = project(j,V)
        # return np.sqrt(np.sum(pj.vector().array()))

    def calc_S(self, U):
        if not self.populated:
            self.populate_arrays(U) 

        S = 0

        q = Function(self.V)
        u = Function(self.V)

        for n in xrange(self.N):
            q.assign(self.Q[n])
            u.assign(self.U[n])

            j = self.j(q)

            a = inner(u,u)*j*dx + self.alpha_sq*(inner(u.dx(0), u.dx(0))/j)*dx
            S += 0.5*assemble(a)*self.dt

        diff = self.Q[-1] - self.qB
        
        err = (1/(2*self.sigma_sq))*assemble(inner(diff,diff)*dx)

        return S + err

            
    def qh_at_t1(self):
        # Find q hat at t = 1
        p = TestFunction(self.V)
        qh1 = TrialFunction(self.V)

        a = inner(p,qh1)*dx
        # fixed this sign:
        L = 1.0/self.sigma_sq * inner(p,self.Q[-1] - self.qB)*dx

        A = assemble(a)
        b = assemble(L)

        qh1 = Function(self.V)
        solve(A,qh1.vector(),b)

        return qh1

    
    def calc_Qh(self):
        qh = self.qh_at_t1() 

        # Find q hat at each time step by stepping backwards in time from qh1
        p = TestFunction(self.V)
        qh_prev = TrialFunction(self.V)
        
        a = inner(p, qh_prev)*dx
        A = assemble(a)

        qh_prev = Function(self.V) # unknown at next timestep

        u = Function(self.V)
        q = Function(self.V)
        
        for n in reversed(xrange(self.N)):
            u.assign(self.U[n])
            q.assign(self.Q[n])
            j = self.j(q)

            #c = 0.5*(inner(u,u)/j - (self.alpha_sq)*inner(u.dx(0),u.dx(0))/j**3)
            c = 0.5*(inner(u,u)/j - (self.alpha_sq)*self.j(u)**2/j**3)

            ## -- CHANGED -> qh.dx(0) to q.dx(0)
            L = inner(p,qh)*dx - inner(c*p.dx(0),q.dx(0))*self.dt*dx
            
            b = assemble(L)

            solve(A, qh_prev.vector(), b)

            qh.assign(qh_prev)

            self.Qh[n].assign(qh)

        
    def calc_dS(self, U):
        if not self.populated:
            self.populate_arrays(U) 
        
        v = TestFunction(self.V)
        dS = TrialFunction(self.V)
        
        a = inner(v,dS)*dx

        A = assemble(a)

        dS = Function(self.V)

        for n in xrange(self.N):
            u = self.U[n]
            qh = self.Qh[n]

            j =  self.j(self.Q[n])

            L = inner(v,u*j)*dx + (self.alpha_sq)*inner(v.dx(0),u.dx(0)/j)*dx - inner(v,qh)*dx
            b = assemble(L)

            solve(A, dS.vector(), b)

            #f = A*dS.vector()
            #mf = Function(self.V, f)

            #self.dS[n].assign(dS)
            self.dS[n].vector()[:] = dS.vector().array()

            # self.print_vector_info("dS",self.dS[n])
            # self.print_vector_info("u",u)
            # self.print_vector_info("qh",qh)
            # print " "

            
            
        return np.reshape(self.coeffs_to_matrix(self.dS), self.vec_size)

    def print_vector_info(self,st, u):
        print st, " --> max: ", np.max(u.vector().array()), " min ", np.min(u.vector().array()), " mean: ", np.mean(u.vector().array())

        

    def need_to_repopulate(self, xk):
        self.populated = False

    def populate_arrays(self, U):
        self.U = self.matrix_to_coeffs(np.reshape(U, self.mat_shape))
        self.calc_Q()
        self.calc_Qh()
        self.populated = True
        

    def coeffs_to_matrix(self, C):
        mat = np.zeros(self.mat_shape)

        for n in xrange(self.N):
            mat[:,n] = 1.0*C[n].vector().array()

        return mat

    def matrix_to_coeffs(self, mat):
        C =  [Function(self.V) for i in xrange(self.N)]

        for n in xrange(self.N):
            C[n].vector()[:] = 1.0*mat[:,n]

        return C

    def plot(self, Q):
        plt.figure()
        plt.axis('equal')

        plt.plot(*self.split_array(Q))

    def plot_no_split(self,Q):
        plt.figure()
        plt.axis('equal')

        plt.plot(Q.vector().array())


    def plot_qAqB(self):
        plt.figure()
        plt.axis('equal')
        plt.plot(*self.split_array(self.qA))
        plt.plot(*self.split_array(self.qB))
        plt.show()

    def plot_steps(self):
        plt.ion()
        plt.figure()

        sorted = self.get_sort_order(self.V)

        #X,Y =  self.split_array(self.qB)
        #plt.plot(X[sorted], Y[sorted])
        plt.plot(*self.split_array(self.qB))

        #X,Y =  self.split_array(self.qA)
        #plt.plot(X[sorted], Y[sorted])
        plt.plot(*self.split_array(self.qA))

        plt.axis('equal')

        #X,Y = self.split_array(self.Q[0])
        #line, = plt.plot(X[sorted], Y[sorted])
        line, = plt.plot(*self.split_array(self.Q[0]))

        plt.xlim(-3,3)
        plt.ylim(-3,3)
        
        for q in self.Q:
            #X,Y = self.split_array(q)

            #line.set_data(X[sorted], Y[sorted])
            line.set_data(*self.split_array(q))
            pytime.sleep(2.0*self.dt)

            plt.draw()
            


    # utility functions
    def split_array(self,q):
        if isinstance(q, np.ndarray):
            x = 1.0*q
        else:
            x = 1.0*q.vector().array()

        X = x[0:np.size(x)/2]
        Y = x[np.size(x)/2: np.size(x)]

        return X,Y

    def get_sort_order(self, fun_space):
        vals = interpolate(Expression(('x[0]','x[0]')), fun_space)
        return np.argsort(self.split_array(vals)[0])


#------------------------

def template_size(M, N):
    return Immersion(M,N).template_size

def S(U, M, N, qA, qB):
    im = Immersion(M, N, qA, qB)
    return im.calc_S(U)

def dS(U, M, N, qA, qB):
    im = Immersion(M, N, qA, qB)
    return im.calc_dS(U)


def minimize(M = 100, N = 20, qA=None, qB=None, U = False):
    # retall option returns vectors at each iteration..
    # does tweaking epsilon keep it from blowing up!?

    #callbacks = 0

    def write_U(U):
        np.savetxt('min_u.txt', U, fmt="%12.6G")
        print "--------> ", np.shape(U)
        #callbacks += 1

    im = Immersion(M, N, qA, qB)

    if U is False:
        U = np.zeros(im.vec_size)

    opt = fmin_l_bfgs_b(S, U, fprime=dS, args=(M,N,qA,qB))
    #opt = fmin_bfgs(S, U, fprime=dS, args=(M,N), callback=write_U, full_output=True)#, epsilon=1e-3)


    m = Immersion(M, N, qA, qB)
    im.calc_S(opt[0])

    return [opt, im]

def U_from_file(filename="min_u.txt"):
    #TODO.. determine M,N from file (reshape pre-save)
    U = np.genfromtxt(filename, unpack=True)
    im = Immersion(100,10)
    im.calc_S(U)
    return im
