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
        mesh = dol.Interval(self.M, 0, 2*np.pi)

        V = dol.FunctionSpace(mesh, 'CG', 1)

        exp = dol.Expression("sin(x[0])")

        self.dolf_sin = dol.interpolate(exp, V)
        self.np_sin = np.sin(np.linspace(0,2*np.pi, self.M + 1))

        

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



    # def test_jacobian(self):
    #     dolf_j = dol.dot(self.dolf_sin.dx(0), self.dolf_sin.dx(0))

    #     print dol.assemble(dolf_j*dol.dx)


    # def test_dolf_derivates(self):
    #     dsin = self.dolf_sin

    #     npt.assert_allclose(-dsin.vector().array(), dsin.dx(0).dx(0).vector().array(), rtol=1e-2, atol=1e-2, \
    #                              err_msg="Second derivative of sin(x) is not -sin(x)")



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


    def test_derivative(self):
        im = Immersion(100,2)
        
        u = dol.Expression(('cos(x[0])','sin(x[0])'))
        u = dol.interpolate(u, im.V)

        U = np.ones(im.mat_shape)

        for n in xrange(im.N):
            U[:,n] = 0.0 * u.vector().array()

        S  = im.calc_S(U)
        
        dSs = im.calc_dS(U)
        dSarr = np.reshape(im.calc_dS(U),im.mat_shape)
        #im.calc_dS(U)
        vdS = 0

        v = dol.Expression(('cos(x[0])','cos(x[0])'))
        
        v = dol.interpolate(v, im.V)


        for dS in im.matrix_to_coeffs(dSarr):
            vdS += dol.assemble(dol.dot(v,dS)*dol.dx)*im.dt



        lims = []
        Ss = []
        Sps = []
        #eps = np.array([10**(-n) for n in np.linspace(0,20,20)])
        eps = 10.**(-sp.arange(10))
        

        for ep in eps:
            im = Immersion(100,2)
            Up = np.zeros(im.mat_shape)
            for n in xrange(im.N):
                Up[:,n] = U[:,n] + ep*v.vector().array()
                
            Sp = im.calc_S(Up)
            Ss.append(S)
            Sps.append(Sp)
            lims.append((Sp - S)/ep)


        for n in xrange(len(eps)):
            print eps[n],"-- \t",lims[n],"\t", vdS, "\tS: ", Ss[n],"\tSp: ",Sps[n]
    

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
