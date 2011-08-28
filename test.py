# Unit testing

import dolfin as dol
import numpy as np
import numpy.testing as npt
import scipy as sp
import matplotlib.pylab as plt

import unittest

from image_deformation import Immersion


class TestCurveOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.M = 100
        self.mesh = dol.Interval(self.M, 0, 2*np.pi)

        V = dol.FunctionSpace(self.mesh, 'CG', 1)

        exp = dol.Expression("sin(x[0])")

        self.dolf_sin = dol.interpolate(exp, V)
        self.np_sin = np.sin(np.linspace(0,2*np.pi, self.M + 1))
        self.np_cos = np.cos(np.linspace(0,2*np.pi, self.M + 1))

        

    def test_sin_values(self):
        df = self.dolf_sin.vector().array()
        # values of sin(x) should be the same when generated in dolfin and numpy
        npt.assert_allclose(self.np_sin, df, rtol=1e-12, atol=1e-15)


    def test_vector_manip(self):
        dolf = self.dolf_sin.vector().array()
        dolf += 1

        self.dolf_sin.vector()[:] = dolf

        self.np_sin += 1

        # test to check shuffling values in and out of UFL form is okay
        npt.assert_allclose(self.np_sin, self.dolf_sin.vector().array(), \
                                rtol=1e-12, atol=1e-15)


    def test_dolfin_integration(self):
        inte_sin = dol.assemble(self.dolf_sin*dol.dx)

        npt.assert_allclose(inte_sin, 0.0, rtol=1e-12, atol=1e-15, \
                                err_msg="Integrated dolfin's sin() from 0-2pi is not zero!")

    def test_l2_norm(self):
        np_norm = np.linalg.norm(self.np_sin)
        d_norm = self.dolf_sin.vector().norm('l2')

        npt.assert_allclose(np_norm, d_norm, rtol=1e-12, atol=1e-15, \
                                err_msg="Dolfin norm and numpy norm are not the same")

        manual_norm = np.sum([n**2 for n in self.np_sin])
        manual_norm = np.sqrt(manual_norm)

        npt.assert_allclose(np_norm, manual_norm, rtol=1e-12, atol=1e-15, \
                                err_msg="Manually computed norm check failed")


    def test_L2_norm(self):
        asbl_norm = dol.sqrt(dol.assemble(dol.inner(self.dolf_sin, self.dolf_sin)*dol.dx))

        # FLAG THIS! tol isn't *that* close!

        # the L2 norm of sin(x) should be sqrt(pi)
        npt.assert_allclose(asbl_norm, np.sqrt(np.pi), rtol=1e-2, atol=1e-2, \
                                err_msg="Assembled L2 norm check failed")



    def test_jacobian(self):
        dolf_j = dol.inner(self.dolf_sin.dx(0), self.dolf_sin.dx(0))

        V = dol.FunctionSpace(self.mesh, 'CG', 1)
        pj = dol.project(dolf_j, V)

        coeffs = pj.vector().array()
        
        dj = np.sqrt(np.sum(coeffs))

        nj = np.sqrt(np.dot(self.np_cos,self.np_cos))
        
        f = self.dolf_sin

        dola = dol.assemble(dol.inner(f,f*dj)*dol.dx)
        npa  = dol.assemble(dol.inner(f,f*nj)*dol.dx)

        # Again.. not totally close!
        npt.assert_allclose(dola, npa, rtol=1e-2, atol=1e-2)

    def test_dolf_derivatives(self):
        fsin = self.dolf_sin

        dfsin = fsin.dx(0)
        
        V = dol.FunctionSpace(self.mesh, 'CG', 1)

        dfsin = dol.project(dfsin, V)

        npt.assert_allclose(self.np_cos, dfsin.vector().array(), rtol=1e-2, atol=1e-2, \
                                 err_msg="derivative of sin(x) is not cos(x)")



    def test_dolfin_coeff_init(self):
        # mesh = dol.Interval(100, 0, 2*dol.pi)

        # V = dol.VectorFunctionSpace(mesh, 'CG', 2, dim=2)

        # Aexp = dol.Expression(('sin(x[0])','cos(x[0])'))
        # A = dol.interpolate(Aexp, V)

        # B = dol.Function(V)
        

        # intrvl = np.linspace(0,2*dol.pi,201)
        # x, y = np.sin(intrvl), np.cos(intrvl)

        # B.vector()[:] = np.append(x,y)

        # Ax, Ay = self.split_xy_array(A)
        # Bx, By = self.split_xy_array(B)

        # plt.figure()
        # plt.plot(Ax)
        # plt.plot(Bx)
        # #ordr = self.get_sort_order(V)        

        # # the interpolated dolfin expression should be a concatination x y vectors
        # # created with numpy
        # npt.assert_allclose(A.vector().array(), B.vector().array(), rtol=1e-10, atol=1e-12, \
        # #npt.assert_allclose(Ax[ordr], Bx[ordr], rtol=1e-10, atol=1e-12, \
        #                         err_msg="Different methods of coeff init failed")
        pass

    def test_template_target_does_not_change(self):
        im = Immersion(100,10)
        
        qA = 1.0*im.qA.vector().array()
        qB = 1.0*im.qA.vector().array()

        U = 0.1*np.ones(im.mat_shape)

        im.calc_S(U)

        npt.assert_allclose(qA, im.qA.vector().array(), rtol=1e-10, atol=1e-12, 
                            err_msg="qA changes after calculating S!!")
        npt.assert_allclose(qB, im.qA.vector().array(), rtol=1e-10, atol=1e-12,
                                err_msg="qA changes after calculating S!!")
        

    def test_matrix_to_coeff_conversion(self):
        im = Immersion(50,50)
        
        U = np.random.rand(*im.mat_shape)

        Uc = im.matrix_to_coeffs(U)

        U2 = im.coeffs_to_matrix(Uc)


        for n in xrange(50):
            npt.assert_allclose(np.sum(U[:,n]), np.sum(Uc[n].vector().array()), 
                                rtol=1e-10, atol=1e-12)
            

        npt.assert_allclose(U, U2, rtol=1e-10, atol=1e-12)

    def test_dS_reshape(self):
        im = Immersion(50,50)

        U = np.zeros(im.mat_shape)

        dSv = im.calc_dS(U)

        dSm = np.reshape(dSv, im.mat_shape)
        
        for n in xrange(50):
            npt.assert_allclose(dSm[:,n],im.dS[n].vector().array(), 
                                rtol=1e-10, atol=1e-12)


    def test_mass_matrix_mult(self):
        im = Immersion(100,10)

        v = dol.TestFunction(im.V)
        u = dol.TrialFunction(im.V)
        
        a = dol.inner(v,u)*dol.dx
        A = dol.assemble(a)

        q = im.qA

        u1 = dol.Function(im.V)
        u2 = dol.Function(im.V)

        g = A*q.vector()
        f = dol.Function(im.V, g)

        u1.assign(f)

        u2.vector()[:] = A * q.vector().array()

        npt.assert_allclose(u1.vector().array(),u2.vector().array(), 
                                rtol=1e-10, atol=1e-12)
        
                     

    def test_derivative(self):
        N = 10
        M = 100

        im = Immersion(M,N)
        
        u = dol.Expression(('cos(x[0])/10.0','cos(x[0])/10.0'))
        #u = dol.Expression(('0.1','0.1'))
        u = dol.interpolate(u, im.V)

        U = np.zeros(im.mat_shape)

        u_scaler = 0.2

        for n in xrange(im.N):
            U[:,n] = u_scaler * u.vector().array()
            
        S  = im.calc_S(U)
        
        dSarr = np.reshape(im.calc_dS(U),im.mat_shape)

        vdS = 0

        #v = dol.Expression(('cos(x[0])/2.0','cos(x[0])/2.0'))
        v = dol.Expression(('pow(x[0],2)/3.0','pow(x[0],2)/3.0'))
        #v = dol.Expression(('x[0]','x[0]'))
        #v = dol.Expression(('x[0]*0.001','x[0]*0.001'))
        v = dol.interpolate(v, im.V)


        s_vdS = 0
        s_v = 0

        for dS in im.matrix_to_coeffs(dSarr):
            #print "--->", dS.vector().array(), v.vector().array()
            #print "--> ", np.sum(dS.vector().array()), np.sum(v.vector().array())

            vdS += dol.assemble(dol.inner(v,dS)*dol.dx)

        #print s_vdS * im.dt
        vdS *= im.dt



        lims = []
        Ss = []
        Sps = []
        eps = 10.**(-sp.arange(20))
        

        for ep in eps:
            im = Immersion(M,N)
            Up = np.zeros(im.mat_shape)
            for n in xrange(im.N):
                Up[:,n] = u_scaler * u.vector().array() + ep*v.vector().array()

                
            Sp = im.calc_S(Up)
            Ss.append(S)
            Sps.append(Sp)
            lims.append((Sp - S)/ep)

        print "%s %15s %12s %12s %15s %15s" % ("Eps","LHS","RHS","S","Sp", "Sp - S")
        for n in xrange(len(eps)):
            print "%.0e  %15.6f  %12.6f  %12.7f  %12.7f %12.7f" % \
                (eps[n], vdS, lims[n], Ss[n], Sps[n], Sps[n] - Ss[n])

    

    # UTILITY FUNCTIONS
    # -----------------

    def get_sort_order(self, fun_space):
        vals = dol.interpolate(dol.Expression(('x[0]','x[0]')), fun_space)
        return np.argsort(self.split_xy_array(vals)[0])


    def split_xy_array(self, q):
        if isinstance(q, np.ndarray):
            x = q
        else:
            x = q.vector().array()

        X = x[0:np.size(x)/2]
        Y = x[np.size(x)/2: np.size(x)]

        return X,Y


if __name__ == '__main__':
    unittest.main()
