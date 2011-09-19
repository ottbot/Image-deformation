
import os, sys
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import glob
import re

import numpy as np

import matplotlib
matplotlib.use("cairo")

import matplotlib.pylab as plt

import image_deformation as img

""" This is the shape match test """

M, N = 100, 10

sigma = 0.001
alpha = 0.001

# try with this hand drawn star
tmpl = np.genfromtxt("shape_match/test_star.txt", unpack=True)


results = []

for shape_file in glob.glob("shape_match/*.txt"):
    if not re.search('test_',shape_file):
        targ = np.genfromtxt(shape_file, unpack=True)

        o = img.minimize(M=M,N=N,qA=tmpl, qB=targ, alpha=alpha, sigma=sigma)

        results.append([[shape_file, o[0][1]], o])

        o[1].plot_steps_held()
        plt.savefig(shape_file.replace(".txt",".pdf"),bbox_inches='tight')

        print shape_file, o[0][1] #, o[0][1]


match = None

for res in results:
    if match is None:
        match = res
    else:
        if res[0][1] < match[0][1]:
            match = res

print "Matching shape is: ", match[0][0]
