# -*- coding: utf-8 -*-
"""
In this class we would like to provide functions which would be useful for the evaluation 
of our model regarding to the prioe knowledge
"""

import numpy as np
import math
class Prior_Analysis(object):
    
    def __init__(self,Distributions,Balanced_Distributions,Naive_Distributions, Latency):
        
        self.Initial_Dists = Distributions
        
        self.Balanced_Dists = Balanced_Distributions
        
        self.Naive_Balanced = Naive_Distributions
        
        self.Latency = Latency
        
        (a , b,c) = np.shape(self.Initial_Dists)
        
        self.W = a
        
        


#####################Transformation Matrix#########################
    def TransformationMatrix(self, kind):
        if kind == 0:
            T = np.dot(self.Initial_Dists[:,:,0],self.Initial_Dists[:,:,1])
        elif kind == 1:
            
            T = np.dot(self.Balanced_Dists[:,:,0],self.Balanced_Dists[:,:,1])
        elif kind == 2:
  
            T = np.dot(self.Naive_Balanced[:,:,0],self.Naive_Balanced[:,:,1])
            
        return T
#Entropy
    
    def Entropy(self,kind):
        T = self.TransformationMatrix(kind)
        H = []
        for i in range(self.W):
            List = []
            for k in range(self.W):
                List.append(T[i,k])
            L =[]
            for item in List:
                if item!=0:
                    L.append(item)
            l = sum(L)
            for i in range(len(L)):
                L[i]=L[i]/l
            ent = 0
            for item in L:
                ent = ent - item*(np.log(item)/np.log(2))
            H.append(ent)
        if sum(H) == 0:
            return 0
        for j in range(len(H)):
            H[j] = H[j]/len(H)
        Entropy = sum(H)
        return Entropy
   
   


#AveragedDelay
    def Ave_Latency(self,kind, mu):
        AV_Delay = []
        if kind == 0:
            Routing = self.Initial_Dists
        elif kind == 1:
            Routing = self.Balanced_Dists
        elif kind ==2:
            Routing = self.Naive_Balanced
            
        for i in range(self.W):
            for j in range(self.W):
                for k in range(self.W):

                     factor = (1/self.W)*Routing[i,j,0]*Routing[j,k,1]
                     Delay = self.Latency[i][j] + self.Latency[j+self.W][k]
                     
                     AV_Delay.append(Delay*factor)
        ADelay = 0
        for I in range(len(AV_Delay)):
            ADelay = ADelay + AV_Delay[I]
                     
        return ADelay
                     
                     
