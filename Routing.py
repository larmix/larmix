# -*- coding: utf-8 -*-
"""
Routing: In this class we aim to make our routing policies materealized 
"""


def I_key_finder(x,y,z):
    import json

    with open('NYM.json','r') as dicts:
        data = json.loads(json.load(dicts))
    
    for i in range(len(data['x'])):
        if int(x*1000) == int(1000*data['x'][i]) and int(y*1000) == int(1000*data['y'][i]) and int(z*1000) == int(1000*data['z'][i]):
            I_key = data['i_key'][i]
    
    return I_key
            
 
import numpy as np
class Routing(object):
    
    def __init__(self,tau,Distances,kind,Hparameter,NYM = False):
        
        self.D = Distances
        (a1,b1,c1) = np.shape(self.D)
        self.L = c1+1
        self.W = a1
        self.N = self.L*self.W

        
        self.T = tau
        self.kind = kind
        

        self.Hparameter = Hparameter # useful when we're eager to make a trade off
        self.NYM = NYM
        
        
    def Rout(self):
        
        if self.kind:
            Dist_matrix = self.InitialFunc()
        else:
            
            Dist_matrix = self.Random_Routing()
        return Dist_matrix
        
        
        
    def Random_Routing(self):   #This function will provide the random distribution
        
        A = np.ones((self.W,self.W,self.L-1))
        
        return A*(1/self.W)
    
        
        
        
  


    def SortDis(self): #We want to make a row of distance sorted acsending
        Dis1 = self.D

        Dis = np.zeros((self.W,self.W,self.L-1))# We save the sorted version in this
        #matrix
        INV=np.zeros((self.W,self.W,self.W,self.L-1))
        #We save the information for sorting the matrix later
        
   
        for i in range(self.W):
            for j in range(self.L-1):
                A = 0
                B = 0
                C = 0
                A = Dis1[i,:,j]
                A = np.matrix(A)
                B = np.zeros((self.W,self.W))
                C = -np.matrix((-np.sort(A)))
                for k in range(self.W):
                    for z in range(self.W):
                        if C[0,z] == A[0,k]:
                            LL = 0
                            for jj in range(k):
                                if B[jj,z]==1:
                                    LL = LL+1
                            if LL == 0:    
                                B[k,z]=1
                                break
                Dis[i,:,j] = np.sort(Dis1[i,:,j], axis = 0)            
                INV[i,:,:,j]= np.transpose(B)
        return Dis , INV



    def Distort(self,Dis1):#In this function we try to dis sort 
        #sorted distributions
        
        Dis2,INV = self.SortDis()
        Dis = np.zeros((self.W,self.W,self.L-1))
        for i in range(self.W):
            for j in range(self.L-1):
                Dis[i,:,j] = np.dot( Dis1[i,:,j],INV[i,:,:,j])
        return Dis
    
    
    
    def InitialDist(self,A):#We materealize our function for making the trade off
        #In this function just for one sorted distribution

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

                E = math.exp(-1)
                R = E**J

                B.append(R)
                A[i] = A[i]**(-T)

                g = A[i]*B[i]

                D.append(g)
            n=sum(D)
            for l in range(len(D)):
                D[l]=D[l]/n

            return D
    
    def InitialFunc(self):# We may apply so called function over all possible
        #rows in this function
        DistFuncs1= np.zeros((self.W,self.W,self.L-1))
        (DIS,INV) = self.SortDis()
        for i in range(self.L-1):
            for j in range(self.W):
                DistFuncs1[j,:,i] = self.InitialDist(DIS[j,:,i])

        DistFuncs = self.Distort(DistFuncs1)
        
        #print(DistFuncs)
        return DistFuncs # This the final distributions

































        