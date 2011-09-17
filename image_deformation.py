from dolfin import *
import time as pytime
import os as os

import numpy as np
import scipy as sp

from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pylab as plt

class Immersion:

    # flag to test if the arrays of U,Q, Qh need to be populated
    # call populate_arrays if so 
    populated = False


    def __init__(self,M=100, N=10, qA=None, qB=None, alpha=0.001, sigma=0.001,deg=2):

        self.N = N
        self.M = M
        self.dt = 1./self.N


        self.mesh = Interval(self.M, 0, 2*pi)
        self.V  = VectorFunctionSpace(self.mesh, 'CG', deg, dim=2)


        self.alpha_sq = alpha**2
        self.sigma_sq = sigma**2


        if qA is None:          # then use a default qA
            self.qA_exp = Expression(('100*sin(x[0])','100*cos(x[0])'))
            self.qA = interpolate(self.qA_exp, self.V)
        else:
            if isinstance(qA,tuple):
                self.qA_exp = Expression(qA)
                self.qA = interpolate(self.qA_exp, self.V)
            else:
                self.qA = Function(self.V)
                self.qA.vector()[:] = qA


        if  qB is None:         # then use a default qB
            self.qB_exp = Expression(('50*sin(x[0])','50*cos(x[0])'))
            self.qB = interpolate(self.qB_exp, self.V)
        else:
            if isinstance(qB,tuple):
                self.qB_exp = Expression(qB)
                self.qB = interpolate(self.qB_exp, self.V)
            else:
                self.qB = Function(self.V)
                self.qB.vector()[:] = qB


        # Determine axis lims for plotting
        minA, maxA = np.min(self.qA.vector().array()), np.max(self.qA.vector().array())
        minB, maxB = np.min(self.qB.vector().array()), np.max(self.qB.vector().array())

        mins = minA if minA < minB else minB
        maxs = maxA if maxA > maxB else maxB

        pad = np.abs((maxs-mins)/6)
        
        lbnd = int(round(mins - pad,-1))
        ubnd = int(round(maxs + pad,-1))

        self.axis_bounds = (lbnd,ubnd,lbnd,ubnd,)
        

        # determine size needed to input/output vectors
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



    "Calculate the functional S for a given velocity, by combining metric and penalty terms"
    def calc_S(self, U):
        if not self.populated:
            self.populate_arrays(U) 

        return self.metric() + self.penalty()
        
    "The metric term"
    def metric(self):
        E = 0

        q = Function(self.V)
        u = Function(self.V)

        for n in xrange(self.N):
            q.assign(self.Q[n])
            u.assign(self.U[n])

            j = self.j(q)

            a = inner(u,u)*j*dx + self.alpha_sq*(inner(u.dx(0), u.dx(0))/j)*dx
            E += 0.5*assemble(a)*self.dt

        return E

    "The penalty term"
    def penalty(self):
        diff = self.Q[-1] - self.qB
        return 1/(2*self.sigma_sq)*assemble(inner(diff,diff)*dx)


    "Calcuate the q hat at t=1"
    def qh_at_t1(self):
        # Find q hat at t = 1
        p = TestFunction(self.V)
        qh1 = TrialFunction(self.V)

        a = inner(p,qh1)*dx
        # NOTE: This L should have opposite sign, but doing so flips the sign
        # of the resulting dSdu.. So there's probably a sign error somewhere else!
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

            c = 0.5*(inner(u,u)/j - (self.alpha_sq)*self.j(u)**2/j**3)

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
            
        return np.reshape(self.coeffs_to_matrix(self.dS), self.vec_size)
        

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

    def new_figure(self):
        f = plt.figure()
        f.subplots_adjust(bottom=0.1,top=0.97,left=0.06,right=0.98)
        plt.axis(self.axis_bounds)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.draw()


    def plot(self, Q):
        self.new_figure()
        plt.plot(*self.split_array(Q))

    def plot_step(self, n):
        self.new_figure()

        plt.plot(*self.split_array(self.qA),ls="--")
        plt.plot(*self.split_array(self.Q[n]),color='r')



    def plot_quiver(self, n):
        self.new_figure()

        x,y = self.split_array(self.Q[n])

        u,v = self.split_array(self.U[n])

        mag = [np.sqrt(u[i]**2+v[i]**2) for i in xrange(np.size(u))]
        norm = plt.normalize(np.min(mag), np.max(mag))

        C = [plt.cm.jet(norm(m)) for m in mag]

        plt.plot(x,y)
        plt.quiver(x,y,-u,-v,color=C)
        #plt.plot(*self.split_array(self.qA),color='grey',ls=':')
        plt.plot(*self.split_array(self.qB),color='grey',ls=':')


    def plot_path(self, sample_step = 1):
        idx = np.arange(0,(self.template_size / 2), sample_step)

        qx, qy = self.split_array(self.qA)
        paths = dict([(i,([qx[i]],[qy[i]])) for i in idx])


        for q in self.Q:
            qx, qy = self.split_array(q)
            for k,v in paths.iteritems():
                v[0].append(qx[k])
                v[1].append(qy[k])
        

        self.new_figure()
        plt.plot(*self.split_array(self.qA),ls='-',lw=2,color='b')

        for k,v in paths.iteritems():
            plt.plot(*v,color='r')

        plt.plot(*self.split_array(self.qB),ls='-',lw=2,color='g')        
        

    def plot_no_split(self,Q):
        plt.figure()

        plt.plot(Q.vector().array())


    def plot_qAqB(self):
        self.new_figure()
        plt.plot(*self.split_array(self.qA))
        plt.plot(*self.split_array(self.qB))


    def plot_steps(self):
        plt.ion()
        self.new_figure()

        plt.plot(*self.split_array(self.qA),ls='--')

        line, = plt.plot(*self.split_array(self.Q[0]),lw=2)

        for q in self.Q:
            qsplt = self.split_array(q)

            plt.plot(*qsplt,ls=':')
            line.set_data(*qsplt)

            pytime.sleep(3.0*self.dt)
            plt.draw()
            

    def plot_steps_held(self):
        self.new_figure()

        plt.plot(*self.split_array(self.qB),ls='-')
        plt.plot(*self.split_array(self.qA),ls='-')

        #plt.plot(*self.split_array(self.Q[0]))

        for q in self.Q:
            plt.plot(*self.split_array(q),ls=':')


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


def template_size(**kwargs):
    return Immersion(**kwargs).template_size

def S(U, *args):
    kwargs = args[0] #hack to get kwargs back out..
    im = Immersion(**kwargs)
    return im.calc_S(U)

def dS(U, *args):
    kwargs = args[0] #hack to get kwargs back out..
    im = Immersion(**kwargs)
    return im.calc_dS(U)


"Run a minimisation, takes the arguments needed to setup an Immersion class"
def minimize(**kwargs):
    im = Immersion(**kwargs)
    U = np.zeros(im.vec_size)

    opt = fmin_l_bfgs_b(S, U, fprime=dS, args=[kwargs]) 

    im = Immersion(**kwargs)
    im.calc_S(opt[0])

    return [opt, im]

"Run a minimisation, takes the arguments needed to setup an Immersion class"
def minimise(**kwargs):
    return minimize(**kwargs)


#------- TODO: move these to external module..

"""
This will run a test case from a directory in cases. 
This must contain qa.txt and qb.txt, and you'll need to pass
M if it differs from the defaults of M=100 (or you'll get an error)
"""
def run_case(casename, **kwargs):
    #read tmpl, targ from files
    case = "cases/" + casename + "/"

    if os.path.exists(case):
        targ = np.genfromtxt(case + "qb.txt", unpack=True)
        tmpl = np.genfromtxt(case + "qa.txt", unpack=True)
        
        o = minimize(qA=tmpl,qB=targ, **kwargs)

        print "case ", case, "S = ", o[0][1]
        im = o[1]

        for n in xrange(im.N):
            im.plot(im.Q[n])
            plt.savefig("%sq_%d.pdf" % (case, n),bbox_inches='tight')

            im.plot_step(n)
            plt.savefig("%sstep_%d.pdf" % (case, n),bbox_inches='tight')

            im.plot_quiver(n)
            plt.savefig("%squiver_%d.pdf" % (case, n),bbox_inches='tight')

            
        im.plot_steps_held()
        plt.savefig(case+"steps.pdf",bbox_inches='tight')

        im.plot(im.qA)
        plt.savefig(case+"qa.pdf",bbox_inches='tight')

        im.plot(im.qB)
        plt.savefig(case+"qb.pdf",bbox_inches='tight')

        im.plot_qAqB()
        plt.savefig(case+"qaqb.pdf",bbox_inches='tight')


        im.plot_path(1)
        plt.savefig(case+"path.pdf",bbox_inches='tight')

        np.savetxt(case+"u.txt", o[0][0], fmt="%12.6G")

        return im
    else:
        if os.path.exists("cases"):
            cases = ", ".join(os.listdir("cases"))
        else:
            cases = "NONE -- no cases directory"
        raise OSError, "That case does not exist. Choices: " + cases


def load_case(casename):
    case = "cases/" + casename + "/"

    if os.path.exists(case+"u.txt"):
        
        targ = np.genfromtxt(case + "qb.txt", unpack=True)
        tmpl = np.genfromtxt(case + "qa.txt", unpack=True)
        U = np.genfromtxt(case + "u.txt", unpack=True)

        im = Immersion(100, 10, tmpl,targ,0.001,0.001)

        im.calc_S(U)

        return im
    else:
        raise OSError, "That case does not exist, or has no been run yet."


def deg_test(degs):
    case = "deg_test"

    ims = []

    for d in degs:
        print d
        targ = np.genfromtxt("%s/%d_qb.txt" % (case,d), unpack=True)
        tmpl = np.genfromtxt("%s/%d_qa.txt" % (case,d), unpack=True)

        print np.shape(tmpl)
        print np.shape(targ)

        opt = minimize(qA=tmpl,qB=targ,deg=d)

        opt[1].plot_steps_held()
        ims.append(opt)

    return ims

def run_reparm(M=100,N=10):
    case = "reparams"

    qa = ('100*sin(x[0])','100*cos(x[0])')
    qb1 = ('50*sin(x[0])','50*cos(x[0])')

    qb2 = ('50*sin(x[0]+pi/4.)','50*cos(x[0]+pi/4.)')
    qb3 = ('50*sin(2*x[0])','50*cos(2*x[0])')
    qb4 = ('50*cos(x[0])','50*sin(x[0])')
    

    opts = []

    for i, qb in enumerate([qb1,qb2,qb3,qb4]):
        o = minimise(M=M, N=N, qA=qa, qB=qb)

        print "Reparam ", i, " S = ",o[0][1]
        o[1].plot_steps_held()
        plt.savefig("%s/steps_%d.pdf" % (case, i),bbox_inches='tight')

        o[1].plot_path(3)
        plt.savefig("%s/path_%d.pdf" % (case, i),bbox_inches='tight')

        opts.append(o)

    return opts


