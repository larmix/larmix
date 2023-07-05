# -*- coding: utf-8 -*-
"""
Fig1
"""

def nCr(n,r):
    import math
    f = math.factorial
    return f(n) // f(r) // f(n-r)
def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


class test(object):
    
    def __init__(self,T,H):
        self.T = T
        self.Hparameter = H

    def InitialDist(self,A,t,r,e):#We materealize our function for making the trade off
        #In this function just for one sorted distribution
        self.Hparameter = r
        self.T = t

        import math
        B=[]
        D=[]
        t = self.T
        if self.T == 0:
            Max = min(A)
            for i in range(len(A)):
                D.append(0)
                if A[i] == Max:
                    index = i
                    D[index] = 1
            return D
        
        else:
            T=1-self.T
            r = self.Hparameter
            for i in range(len(A)):
                j = i
                J = (j*(1/(t**(r))))**(1-t)

                E = 1/(math.exp(1))
                R = E**J

                B.append(R)
                A[i] = A[i]**(-T)

                g = A[i]*B[i]

                D.append(g)
            n=sum(D)
            for l in range(len(D)):
                D[l]=D[l]/n

            return D
        


    def kUkullback_leibler(self,Dist):
        import numpy as np 
        
        m = len(Dist)
        output = 0
        for item in Dist:
            if not item ==0:
                output = output + item*(np.log(item)/np.log(2)) 
                output = output + item*(np.log(m)/np.log(2)) 
        return output
