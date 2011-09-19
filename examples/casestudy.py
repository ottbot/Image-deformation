import os, sys
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

from image_deformation import *


""" 
This module is used to (re)generate case studies used in the report
"""


def run_case(casename, **kwargs):
    """
    This will run a test case from a directory in cases. 
    This must contain qa.txt and qb.txt, and you'll need to pass
    M if it differs from the defaults of M=100 (or you'll get an error)

    casename is the name of a directory in ./cases that contain a qa.txt 
    and qb.txt

    """

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
    """ Load a previously run case, with out running it again """
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
    """ The optimiser with different degrees of FEM polynomials """
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
    """ Run the reparameterision cases """
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



