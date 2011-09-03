import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

class Vectorpng:

    def __init__(self, filename, N=402):
        self.N = N/2.
        self.im = plt.imread(filename)

        (self.H,self.W,self.A) = self.im.shape

        self.X = [] #np.array([])
        self.Y = [] #np.array([])

        self.outline()

    def is_white(self, px):
        return np.unique(px[0:3])[0] == 1.0
        # if px.size is 3:
        #     return (px.sum() ==  3.0)
        # else:
        #     if px[3]:
        #         return (px.sum() ==  3.0)
        #     else:
        #         return True

    def is_black(self,px):
        if isinstance(px, tuple):
            px = self.im[px[1],px[0]]

        if np.size(px) == 4:
            return px[3] and np.unique(px[0:3])[0] == 0.0
        else:
            return np.unique(px[0:3])[0] == 0.0


    def find_seed(self):
        # find a place to start


        rad = 5 # how much to insure we're on a real black spot

        tries = 0
        maxtries = 5000

        while tries < maxtries:

            # try a random pixel, far from the borders
            lim = self.H if self.H < self.W else self.W

            x = np.random.random_integers(2*rad,lim - 2*rad)
            y = np.random.random_integers(2*rad,lim - 2*rad)

            px = self.im[y,x]

            if self.is_black(px):
                badseed = 0

                badseed += sum([int(self.is_black((y,i))) for i in xrange(x,x+rad)])
                badseed += sum([int(self.is_black((y,i))) for i in xrange(x-rad,x)])

                badseed += sum([int(self.is_black((i,x))) for i in xrange(y,y+rad)])
                badseed += sum([int(self.is_black((i,x))) for i in xrange(y-rad,y)])



                if not badseed:
                    return x,y
                

            tries +=1    

                             

        raise ValueError, \
            "Max tries to find seed pixel.. Are you sure this has a black blob?"
                            
            

    def append(self,x,y):
        self.X.append(x)
        self.Y.append(y)

        
    def in_bounds(self, x,y):
        return (0 <= x < self.W) and (0 <= y < self.H)

    
    def move_clockwise(self, c,p):
        x,y = c

        #print x, y
        d = 5
        dx, dy = (p[0] - x, p[1] - y)

        if dx is -1: 
            if dy is 1:
                return (x+1,y) #move right
            else: 
                return (x,y-1) #move up
        if dx is 0:
            if dy is 1:
                return (x-1,y) #move left
            else:
                return (x+1,y) #move right
        if dx is 1:
            if dy is -1:
                return (x-1,y) #move left
            else:
                return (x,y+1) #move down

        print "we're NONE of those!", dx, dy
        return p
        #print "we're NONE of those!", dx, dy


    def outline(self):
        sx,sy = self.find_seed()
        
        #sx, sy = 140, 140

        #from seed, search left to until non black..
        for i in reversed(xrange(0,sx)):
            c = i,sy
            if not self.is_black(c):
                x = i
                y = sy
                
                #self.append(x,y)
                break
            else:
                p = c
        
        at_start = False

        #y = sx

        c1 = c
        p1 = p
        self.append(*p)

        im = self.im

        n = 0
        maxn = 100
        


        bck = c

        while n < maxn:

            if self.is_black(p):
                print "P is bk"
            else:
                print "P is not bk"

            if not self.in_bounds(*c):
                print "we;re not bounds!", c

            if self.is_black(c):
                self.append(*c)
                p = c
                c = bck

            else:
                bck = c
                c = self.move_clockwise(c,p)
            

            n += 1
            #xn,yn = mv_up(x,y)
            #if (xn,yn) == (x,y):
            #    self.append(xn,yn)
            #x = xn
            #y = yn - 1

            # l =  im[y,x-1] 
            # if self.is_black(l):
            #     x -= 1
            # else:
            #     ul = im[y-1,x-1]
            #     if  self.is_black(ul):
            #         x -= 1
            #         y -= 1
            #     else:
            #         u = im[y-1,x]
            #         if self.is_black(u):
            #             y -= 1
                    
            #         else:
            #             ur = im[y-1,x+1]
            #             if  self.is_black(ur):
            #                 y -= 1
            #                 x += 1
            #             else:
            #                 r = im[y,x+1]
            #                 if self.is_black(r):
            #                     x += 1
            #                 else:
            #                     dr = im[y+1,x+1]
            #                     if self.is_black(dr):
            #                         y += 1
            #                         x += 1
            #                     else:
            #                         d = im[y+1,x]
            #                         if self.is_black(d):
            #                             y += 1
            #                         else:
            #                             dl = im[y+1,x-1]
            #                             if self.is_black(dl):
            #                                 print "hit a DL!"
            #                                 x -= 1
            #                                 y += 1

            #                             else:
            #                                 print n, "no where to go"
            #                                 no_app = True
                                            
                                            
            
                                            
            # if not old_xy == (x,y):
            #     self.append(x,y)            

            # n += 1
            
        


        # search clock wise until we get back to seed (or near)


    def xy(self):
        return self.X, self.Y
        #return (np.array(self.X2), np.array(self.Y2))




    def interp(self, x, y):
        xrang = np.linspace(x.min(), x.max(), np.size(x))
        f = interp1d(xrang, y)
        xrang = np.linspace(x.min(), x.max(), self.N)

        return f(xrang)

    #http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/
    def smooth_gaussian(self, arr, degree=5):  
         window=degree*2-1  
         weight=np.array([1.0]*window)  
         weightGauss=[]  
         for i in range(window):  
             i=i-degree+1  
             frac=i/float(window)  
             gauss=1/(np.exp((4*(frac))**2))  
             weightGauss.append(gauss)  
         weight=np.array(weightGauss)*weight  
         smoothed=[0.0]*(len(arr)-window)  
         for i in range(len(smoothed)):  
             smoothed[i]=sum(np.array(arr[i:i+window])*weight)/sum(weight)  
         return smoothed 



v = Vectorpng("circle.png")

X,Y = v.xy()

# srt = np.argsort(X)
# X = X[srt]
# Y = Y[srt]


plt.figure()
plt.imshow(v.im)
plt.plot(X,Y,':')






        # todo.. check is png is valid

        # vert pass (get X)
        # for i in xrange(self.H):
        #     self.on_black = False
        #     for j in xrange(self.W):
        #         if self.fuzz:
        #             if not self.in_fuzz:
        #                 self.px_add(i,j,'X')
        #             else:
        #                 self.in_fuzz -= 1
        #         else:
        #             self.px_add(i,j)

        # horiz pass (get Y)
    #     for j in xrange(self.W):
    #         self.on_black = False
    #         for i in xrange(self.H):
    #             if self.fuzz:
    #                 if not self.in_fuzz:
    #                     self.px_add(i,j)
    #                 else:
    #                     self.in_fuzz -= 1
    #             else:
    #                 self.px_add(i,j)


    #     self.X2.reverse()

    #     X = np.concatenate((self.X1, self.X2))
    #     Y = np.concatenate((self.Y1, self.Y2))

    #     X = self.smooth_gaussian(X)
    #     Y = self.smooth_gaussian(Y)

    #     X = np.append(X, X[0])
    #     Y = np.append(Y, Y[0])

    #     self.X = self.interp(np.array(Y),np.array(X))
    #     self.Y = self.interp(np.array(X),np.array(Y))



    # def px_add(self, i,j):
    #     px = self.im[self.H - 1 - i,j]
    #     if self.on_black:
    #         if self.is_white(px):
    #             self.X1.append(j)
    #             self.Y1.append(i)

    #             self.on_black = False
    #             self.in_fuzz = self.fuzz
    #     else:
    #         if not self.is_white(px):
    #             self.X2.append(j)
    #             self.Y2.append(i)

    #             self.on_black = True
    #             self.in_fuzz = self.fuzz
