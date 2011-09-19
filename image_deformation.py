from dolfin import *
import time as pytime
import os as os

import numpy as np
import scipy as sp

from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pylab as plt


class Immersion:
    """ Performs a deformation of an image given a velocity field """

    # flag to test if the arrays of U,Q, Qh need to be populated
    # call populate_arrays if so. This is used so S and dS can be calculated independently
    populated = False


    def __init__(self,M=100, N=10, qA=None, qB=None, alpha=0.001, sigma=0.001,deg=2):
        """ Initializes a new immersion class, with named arguments

        M: Number of spatial interval cells
        N: Number of time steps
        qA: Array of the template shape (x and y concatenated), 
              *or* a tuple containing a string expression for x and y dims.
        qB: Array of the target shape (x and y concatenated), 
              *or* a tuple containing a string expression for x and y dims.
        alpha: The alpha constant
        sigma: Yep, it's the the sigma constant
        deg: The degree of the Lagrange finite elements polynomial


        ===========
        After initialising an Immersion class, you pass the velocity when
        calling the calc_S() or calc_dS() methods -- either of which will
        calculate Q


        These the same arguments are used to run the optimisation problem, you can
        pass them to the module method minimize() to find the U to give the miminum S 
        and geodesic Q.

        In most cases, you don't actually initialize an Immersion class, just use the minimize() 
        method
        """
        

        self.N = N
        self.M = M
        self.dt = 1./self.N


        # Interval from 0 to 2pi, divided into M cells
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
        """Calculates the progression of q after the velocity field has been set."""

        r = TestFunction(self.V)
        q_next = TrialFunction(self.V)

        a = inner(r,q_next)*dx
        A = assemble(a)        

        q_next = Function(self.V)   # the unknown at a new time level
        q = Function(self.V)

        #initial q at t=0 is qA
        q.assign(self.qA)

        for n in xrange(self.N):
            L = inner(q, r)*dx -  self.dt*inner(r,self.U[n])*dx
            b = assemble(L)

            solve(A, q_next.vector(), b)

            q.assign(q_next)

            self.Q[n].assign(q)


    def j(self, q):
        """ Gives |dq/ds| """
        return  sqrt(inner(q.dx(0),q.dx(0)))




    def calc_S(self, U):
        """
        Calculate the functional S for a given velocity U
        by combining metric and penalty terms
        """
        if not self.populated:
            self.populate_arrays(U) 

        return self.metric() + self.penalty()
        

    def metric(self):
        """ The metric term of S"""
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


    def penalty(self):
        """The penalty term, or matching functional, of S."""
        diff = self.Q[-1] - self.qB
        return 1/(2*self.sigma_sq)*assemble(inner(diff,diff)*dx)



    def qh_at_t1(self):
        """Calcuates the q hat at t=1"""     
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
        """ Solve q hat at each timestep """
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
        """ Find dS/du, the gradient of S, at each time step for a given velocity """

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
        

    def populate_arrays(self, U):
        """ 
        Convert the inputted matrix/vector of velocity U to UFL form, 
        and calculate q and q hat at each time step.

        This happens when calc_S() or calc_dS() is called (usually at
        each iteration of the optimizer)
        """
        self.U = self.matrix_to_coeffs(np.reshape(U, self.mat_shape))
        self.calc_Q()
        self.calc_Qh()
        self.populated = True
        

    def coeffs_to_matrix(self, C):
        """ 
        Convert an array representing a  UFL coefficient form
        at each time step to a numpy matrix (each column is a vector
        of the values at a particular timestep)
        """

        mat = np.zeros(self.mat_shape)

        for n in xrange(self.N):
            mat[:,n] = 1.0*C[n].vector().array()

        return mat

    def matrix_to_coeffs(self, mat):
        """ 
        Convert a numpy matrix (each column is a vector
        of the values at a particular timestep) to an array representing 
        a UFL coefficient form at each timestep
        """

        C =  [Function(self.V) for i in xrange(self.N)]

        for n in xrange(self.N):
            C[n].vector()[:] = 1.0*mat[:,n]

        return C

    #-------------------- Plotting utils


    def new_figure(self):
        """ New figure with precalcuated axis bounds and aspect 1"""
        f = plt.figure()
        f.subplots_adjust(bottom=0.1,top=0.97,left=0.06,right=0.98)
        plt.axis(self.axis_bounds)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.draw()


    def plot(self, Q):
        """ 
        Plot a single curve q, or anything else, because
        this splits the curve into x and y, then does plot(x,y)
        it's probably only useful for curves
        """
        self.new_figure()
        plt.plot(*self.split_array(Q))

    def plot_step(self, n):
        """ Plot the curve at a particular time step number """
        self.new_figure()

        plt.plot(*self.split_array(self.qA),ls="--")
        plt.plot(*self.split_array(self.Q[n]),color='r')



    def plot_quiver(self, n):
        """ Plot the curve and the vector field at a particular timestep """
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
        """ 
        Plot the path of some points along the curve evolution.
        Parameter controls to number of points to plot
        """
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
        """ Plot a UFL coefficient form, without spliting into to separate dims """
        plt.figure()

        plt.plot(Q.vector().array())


    def plot_qAqB(self):
        """ Plot the template and target on the same figure """
        self.new_figure()
        plt.plot(*self.split_array(self.qA))
        plt.plot(*self.split_array(self.qB))


    def plot_steps(self):
        """ Show an animatation of the evoluation of the deformation """
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
        """ Plot the deformation evolution """
        self.new_figure()

        plt.plot(*self.split_array(self.qB),ls='-')
        plt.plot(*self.split_array(self.qA),ls='-')

        #plt.plot(*self.split_array(self.Q[0]))

        for q in self.Q:
            plt.plot(*self.split_array(q),ls=':')


    # utility functions
    def split_array(self,q):
        """ 
        Split a numpy array, or UFL coefficient form into X, Y numpy
        vectors.
        """
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
    """ 
    Return the size of the template/target vector needed 
    for M cells and the FEM order 
    """
    return Immersion(**kwargs).template_size

def S(U, *args):
    """ 
    Calculate S for a velocity U and  Immersion class params.
    This is called by the BFGS optimiser
    """

    kwargs = args[0] #hack to get kwargs back out..
    im = Immersion(**kwargs)
    return im.calc_S(U)

def dS(U, *args):
    """ 
    Calculate dSdu for a velocity U and  Immersion class params.
    This is called by the BFGS optimiser
    """

    kwargs = args[0] #hack to get kwargs back out..
    im = Immersion(**kwargs)
    return im.calc_dS(U)



def minimize(**kwargs):
    "Run the optimiser. Takes same arguments needed to setup an Immersion class"
    im = Immersion(**kwargs)
    U = np.zeros(im.vec_size)

    opt = fmin_l_bfgs_b(S, U, fprime=dS, args=[kwargs]) 

    im = Immersion(**kwargs)
    im.calc_S(opt[0])

    return [opt, im]
