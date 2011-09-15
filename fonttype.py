import glob
import re
import os

import numpy as np

import matplotlib
matplotlib.use("cairo")

import matplotlib.pylab as plt

import image_deformation as img

M, N = 100, 10

sigma = 0.001
alpha = 0.001

# try with this hand drawn star


font_dir = "font_samples/N/"

letters = glob.glob(font_dir+"*.txt")

#np.random.shuffle(letters)

#test_letter_file = letters.pop()

for test_letter_file in letters:


    results = []

    test_letter = np.genfromtxt(test_letter_file,unpack=True)
    font_name = test_letter_file.replace('.txt','')


    # mkdir to store plots
    if not os.path.exists(font_name):
        os.mkdir(font_name)

    sans_S = []
    serif_S = []


    fout = file(font_name + "/results.txt", 'w')

    other_letters = [fn for fn in letters if not fn == test_letter_file]

    for letter_file in other_letters:
    
    
        letter = np.genfromtxt(letter_file, unpack=True)

        o = img.minimize(M,N,test_letter, letter, alpha, sigma)


        fout.write("%s ----- S=%10.5f\n" %(letter_file,o[0][1]))

        if re.search("sans-",letter_file):
            sans_S.append(o[0][1])
        else:
            serif_S.append(o[0][1])

        o[1].plot_steps_held()

        pdfname = letter_file.replace(".txt",".pdf")
        
        
        plt.savefig(font_name + "/" + pdfname.split('/').pop(),bbox_inches='tight')



    sans_mean = np.mean(sans_S)
    serif_mean = np.mean(serif_S)
    #sans_mean = np.mean([S for S in sans_S if not S == max(sans_S) and not S == min(sans_S)])
    #serif_mean = np.mean([S for S in serif_S if not S == max(serif_S) and not S == min(serif_S)])



    print test_letter_file, "is a:",
    if sans_mean < serif_mean:
        fout.write("Font is is sans-serif ")
        print "SANS-SERIF font!"
    else:
        fout.write(".. is serif ")
        print "SERIF font!"

        fout.write("Mean Sans = %10.4f Mean Serif = %10.4f\n" % (sans_mean, serif_mean))
        fout.close()



