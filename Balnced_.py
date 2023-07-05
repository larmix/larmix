
# -*- coding: utf-8 -*-
"""
Balnced Layers: is a class with different algorithms which have been considered to make the mixing
Layers balanced
"""

import numpy as np
class Balanced_Layers(object):
    
    def __init__(self,Imbalanced_Dist,Decimal_Percision,Algorithm):
        
        self.IMD = Imbalanced_Dist
        (a,b) = np.shape(self.IMD)   #Imbalanced distribution     
        self.SIMD = []#Imbalanced distribution
        self.W = a

                
        for i in range(self.W):
                self.SIMD.append(self.IMD[i,:].tolist())

        self.DP = Decimal_Percision #How accurate you wopuld like to be when 
        #it comes to balace measurment
        
        self.Algorithm = Algorithm
    
    
    def make_the_layer_balanced(self):
        self.Itterations()
        

        
        self.Naive = 0 #self.Swift_Balancing()
        


            
            
    
    
    
    
    
    def CheckBalance(self,D):
        A = True #shows if D is a balanced matrix or not
        I = (10**self.DP)*np.ones((1,self.W))/self.W #Reference matrix

        J=np.matrix(np.average(D,axis = 0))  #Average of the D over colomuns
        for i in range(self.W):

            if round((10**self.DP)*J[0,i]) != round(I[0,i]):
                A = False
                break
        return A 
    
    
    
    
    def Over_Under_Balance_Loaded(self,Matrix):#This function receive a matrix of distribution
        #And assign -1, 0 and 1 to mix nodes which are respectively under loaded, balanced and over lo0aded
        
        LIST = []
        (a,b) = np.shape(Matrix)
        
        for I in range(b):
            factor = 0
            for J in range(a):
                factor = factor + Matrix[J,I]
            LIST.append(factor)
                



        index = []
        for item in LIST:

            
            if (1-10**(-self.DP)< item < 1+10**(-self.DP)):
                
                index.append(0)
            else:
                
                if 1+10**(-self.DP)< item:
                    
                    index.append(1)
                else:
                    
                    index.append(-1)
        return index , LIST
                
            
    def Greedy_algorithm(self):# This function make our greedy algorithm materealized
        index , Sum = self.Over_Under_Balance_Loaded(self.IMD)
        


        
        I_Sum = []
        for item in Sum:
            if item == 0:
                I_Sum.append(0)
            else:
                I_Sum.append(1/item)
    
        Make_it_Balanced = self.IMD
        P = []
                
        for j in range(len(index)):
            PP = []
            for k in range(len(index)):
                if index[k]==-1:
                    PP.append(Make_it_Balanced[j,k])
            S = sum(PP)
            for i in range(len(PP)):
                
                
                if S == 0 :
                    if len(PP)!=0:
                        PP = [1/len(PP)]*len(PP)
                    
 
                else:
                    PP[i] = PP[i]/S

            P.append(PP)
   

        for i in range(len(index)):
            if index[i] == 1:
                
                     
                
                a = (1-I_Sum[i])*Make_it_Balanced[:,i]
                
                Make_it_Balanced[:,i] = I_Sum[i]*Make_it_Balanced[:,i]
                
                for j in range(self.W):
                    J = 0
                    for k in range(self.W):
                        if (index[k]==-1):
                            Make_it_Balanced[j,k] = Make_it_Balanced[j,k] + a[j]*P[j][J]
                            
                            J = J + 1
                            

        return Make_it_Balanced                    
        
    
    def Itterations(self): #Itteration we need 
        Balance = False
        
        while not Balance:
            self.IMD = self.Greedy_algorithm()
            
            Balance = self.CheckBalance(self.IMD)
            
    def Swift_Balancing(self):#The naive solution
        Imbalanced =  np.matrix(self.SIMD)

        index , Sum = self.Over_Under_Balance_Loaded(Imbalanced)
        

        P = []
        
        for j in range(len(index)):
            if index[j] == -1:
                P.append(1-np.sum(Imbalanced[:,j]))
        S = sum(P)
        for I in range(len(P)):
            P[I]= P[I]/S

        a = 0
        for i in range(len(index)):
            if index[i]==1:
                
                a = a + (1-1/Sum[i])*Imbalanced[:,i]
                
                Imbalanced[:,i] = (1/Sum[i])*Imbalanced[:,i] 
      
        J = 0       
        for k in range(len(index)):
            if index[k]==-1:
                Imbalanced[:,k] = Imbalanced[:,k] + a*P[J]
                J = J + 1
     
        return Imbalanced
        
                
                

                    


    
  