# -*- coding: utf-8 -*-
"""
CLAPS
"""


from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import numpy as np

from math import exp
from scipy import constants
def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


class LARMix_VS_CLAPS(object):
    import time
    
    def __init__(self,Latency,Distance,Routing,Tau,X ):

        
        self.Latency = Latency
        self.W = int(len(Latency)/2)
        self.Distance = Distance
        self.Routing = Routing
        
        self.l = 3
        if Tau == 0:
            self.tau = 0.01
            
        else:
            self.tau = Tau
        
        self.L_Bound = (1/self.W)**(1/self.tau)
        self.U_Bound = (1/self.W)**(self.tau)
        self.X = X
        self.DC = 5
        
        self.Algorithm = 'Greedy'

        

    
    def CLAPS(self,Theta,Version = False):
        
        if Version:
            self.L_Bound = 0
            self.U_Bound = (Theta/self.W)
        
        model = LpProblem(name="small-problem")
        Variables = []
        for k in range(self.l-1):
            for i in range(1+k*self.W,(k+1)*self.W+1):
                for j in range((k+1)*self.W+1,(k+2)*self.W+1):
                    Variables.append(LpVariable(name= 'S'+ str(i)+'_'+str(j), lowBound= self.L_Bound,upBound= self.U_Bound))
        obj = []

        for item in self.Latency:
            for term in item:
                obj.append(term)
        
        


        Dis1 = []
        Dis2 = []



        for i in range(2*self.W):
            Item = [[0]*self.W]*(2*self.W)
    
            Item[i] = [1]*self.W                     
            Dis1.append(Item)


        Dis_ = []

        Dis__ = []

        for k in range(self.W):
            Dis__.append([0]*self.W)


        for i in range(self.W): 
            dis_ = []
            for j in range(self.W):
                if i==j:
            
                    dis_.append(1)
            
                else:
            
                    dis_.append(0)
    
            Dis_.append([dis_]*self.W)
        
    
        for counter in range(self.W):
            Dis2.append(Dis_[counter]+Dis__)
            Dis2.append(Dis__+Dis_[counter])
    

        Dis3 = [[]]


        bound_inq1 = 0
        
        for i in range(2*self.W):
            for j in range(self.W):

                bound_inq1 = bound_inq1 + self.Latency[i][j]
                Dis3[0].append(self.Latency[i][j])
            
           
    
    
        bound_inq1 = bound_inq1*self.X



        Co_Ineq = Dis3

        Bound_Ineq = [bound_inq1/self.W]

        Co_Eq = []

        for item1 in Dis1+Dis2:
            A = []
            for item2 in item1:
                for item3 in item2:
                    A.append(item3)
            
            Co_Eq.append(A)

 
        Equations = []
        for j in range(len(Co_Eq)):
            x = 0
            for i in range(2*self.W**2):
                x = x + Variables[i]*Co_Eq[j][i]
            Equations.append(x)
    
        In_equalities = []
        


        for j in range(len(Co_Ineq)): 
            x = 0
            for i in range(2*self.W**2):
                x = x + Variables[i]*Co_Ineq[j][i]
            In_equalities.append(x)
        
# Add the constraints to the model
        Eqs = 1   
    
        for item in Equations:
    

            model += (item  == 1, "constraint" +str(Eqs))
    
            Eqs += 1


        Ins = 1
        


        for term in  In_equalities:
    
            model += (term  <= Bound_Ineq[Ins-1], "constraint_Inq" +str(Ins))
    
            Ins += 1


        Goal = []

        for i in range (len(Variables)):
            Goal.append(obj[i]*Variables[i])


    # Add the objective function to the model
        model += lpSum(Goal)

# Solve the problem
        status = model.solve()
        
        Vars = {}
        for var in model.variables():
            Vars[var.name] = var.value()


        Constraints = {}
        for name, constraint in model.constraints.items():
            Constraints[name] = constraint.value()
            
            
        Status = {}
        
        Status['Goal'] = model.objective.value()
        
        Status['State'] = LpStatus[model.status]

        
        return Vars,Constraints,Status
    
    
    def CLAPS_(self,TH,State):
        import time
        Start = time.time()
        Weights, Constraints, Status = self.CLAPS(TH,State)
        End = time.time()
        
        self.CLAPS_time = End-Start

        import numpy as np
        
        S1 = np.zeros((self.W,self.W))
        S2 = np.zeros((self.W,self.W))  
        
        if  Status['State'] == 'Optimal':
            for i in range(self.W):
                for j in range(self.W):
                    S1[i,j] = Weights['S'+str(i+1)+'_'+str(1+j+self.W)]
                    S2[i,j] = Weights['S'+str(i+1+self.W)+'_'+str(1+j+2*self.W)]  
            S = np.dot(S1,S2)

            
        
        else:
            S = 'None' 
        if not S == 'None':
            
            Entropy_ = self.Entropy(S)
            Latency_ = self.Ave_Latency(S1, S2)
            
            return Entropy_, Latency_
        else:
            
            return 'None','None'

                    
                    

    def Entropy(self,T):
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
    def Ave_Latency(self,S1,S2):
        AV_Delay = []

        for i in range(self.W):
            for j in range(self.W):
                for k in range(self.W):

                     factor = (1/self.W)*S1[i,j]*S2[j,k]
                     Delay =  self.Latency[i][j] + self.Latency[j+self.W][k]
                     
                     AV_Delay.append(Delay*factor)
        ADelay = 0
        for I in range(len(AV_Delay)):
            ADelay = ADelay + AV_Delay[I]
                     
        return ADelay



    def LARMix(self):
        from Routing import Routing
        from Balnced import Balanced_Layers
        import time
        

        Start = time.time()
        R = Routing(self.tau,self.Distance,self.Routing,1)
        
        Initial_Dist = R.Rout()
        End = time.time()
        S1 = Initial_Dist[:,:,0]
        S2 = Initial_Dist[:,:,1]    

     
        Balance1 = Balanced_Layers(S1,self.DC,self.Algorithm)
        
        Balance2 = Balanced_Layers(S2,self.DC,self.Algorithm)
            
        Balance1.make_the_layer_balanced()
            
        Balance2.make_the_layer_balanced()
        
        
        self.LARMIX_time = [(End-Start + Balance1.G_time+ Balance2.G_time),(End-Start + Balance1.N_time+ Balance2.N_time)]
        
        S1_G = Balance1.IMD
        
        S2_G = Balance2.IMD

        
        S1_N = Balance1.Naive
        
        S2_N = Balance2.Naive     
        
        H_L = {}
        
        import numpy as np
        H_L['H_Naive'] = self.Entropy(np.dot(S1_N,S2_N))
        H_L['H_Greedy'] = self.Entropy(np.dot(S1_G,S2_G))        
        
        H_L['L_Naive'] = self.Ave_Latency(S1_N,S2_N)
        H_L['L_Greedy'] = self.Ave_Latency(S1_G,S2_G) 
        
        
        return H_L

        
        
        
        
        
        
        
        
        
        
        
        
        
    
from Clustering import Clustering
from MixNetArrangment import Mix_Arrangements
from Latency import Latency_and_Distance

        
from Corruption import corruptedMix    
    
    
    
from Datasets import Dataset

from Bridge import Bridge

from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle

from Message import message

from Mix_Node import Mix

from Mix_net import MixNet

from Message_Genartion_and_mix_net_processing import Message_Genartion_and_mix_net_processing
import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def Ent(List):
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
    return ent

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

import numpy as np

import pickle
  
from Simulation import Simulation
'''

data = 'RIPE'

Dimention = 128*3
Goal = 'CLAPS'
DF  = Dataset(data,Dimention,Goal)


Data_set =  DF.data_set()


DF.plt_data()



C_M = 'kmedoids'
Routing = True
Balancing = True
N = Dimention

num_Clusters = 2

r = 1

Tau = 0.001

Algorithm = 'Greedy'


Diversify = 0

speed_Function = 'Verloc'

Decimal_precision = 5

Layers = 3

mu = 0.02

Lambda = 0.0001

Capacity = 10000000000000000000000000000000000000000000000000000000000000000

H_N = round(N/3)

rate = 15

num_targets = 200

Iterations = 2

run_time = 0.32

NYM = False
RIPE = True

strategy = 0

frac = 1    
    
    
    
T = [1,1,1]    

Th = [1.25,2.5,5]

L__0 = []
L__1 = []
L__2 = []
E__0 = []
T__0 = []
    
E__1 = []
T__1 = []
E__2 = []
T__2 = []

ITERATION = 1
for Itr in range(ITERATION):    
    

    clusters = Clustering(Data_set,C_M,num_Clusters,3,0)

    clusters.Data_plt()
    Class_CNs = corruptedMix(Data_set,0.3,clusters.Mixes,clusters.Labels,strategy)
    Class_CNs.corrupted_mix_nodes()
    Corruption = Class_CNs.CNs        
    arrangment = Mix_Arrangements(clusters.Mixes,Diversify,clusters.Labels,clusters.Centers,Corruption,frac,True)
        
    arrangment.Topology_plt()
            
    Corruption_New = arrangment.mapping()
        
    LatencyDistance = Latency_and_Distance(arrangment.Topology,speed_Function,Goal,NYM,RIPE)

    Distances = LatencyDistance.Topology_to_Distance()
    Latency = LatencyDistance.RIPE_Latency
                    

    latency = [[],[],[]] 
    entropy = [[],[],[]]  
    time    = [[],[],[]] 
    
    I = 0
              

    for Tau in T:
        
        Claps = LARMix_VS_CLAPS(Latency,Distances,True, Tau,1)
        

        E,L = Claps.CLAPS_(Th[I],True)
        I = I+1
        latency[2].append(L)
        entropy[2].append(E)

        H_L = Claps.LARMix()
    
        latency[0].append(H_L['L_Greedy'])
        latency[1].append(H_L['L_Naive'])    
    
        entropy[0].append(H_L['H_Greedy'])
        entropy[1].append(H_L['H_Naive']) 
        time[2].append(Claps.CLAPS_time)
        time[1].append(Claps.LARMIX_time[1])    
        time[0].append(Claps.LARMIX_time[0])  
        
    T__0.append(time[0])
    L__0.append(latency[0])    
    E__0.append(entropy[0])    
    T__1.append(time[1])
    L__1.append(latency[1])    
    E__1.append(entropy[1])    
    T__2.append(time[2])
    L__2.append(latency[2])    
    E__2.append(entropy[2])    
    
import numpy as np    
    
t_0 = Med(np.transpose(np.matrix(T__0)).tolist())
t_1 = Med(np.transpose(np.matrix(T__1)).tolist())
t_2 = Med(np.transpose(np.matrix(T__2)).tolist())    

e_0 = Med(np.transpose(np.matrix(E__0)).tolist())
e_1 = Med(np.transpose(np.matrix(E__1)).tolist())
e_2 = Med(np.transpose(np.matrix(E__2)).tolist())


l_0 = Med(np.transpose(np.matrix(L__0)).tolist())
l_1 = Med(np.transpose(np.matrix(L__1)).tolist())
l_2 =Med(np.transpose(np.matrix(L__2)).tolist())

        
print(e_0,e_1,e_2,l_0,l_1,l_2)

File_name = 'CLAPS' + str(N)

Times = [t_0,t_1,t_2]

Entropies = [e_0,e_1,e_2]

Latencies = [l_0,l_1,l_2]


Data_CLAPS = {}

Data_CLAPS['Time'] = Times
Data_CLAPS['Entropy'] = Entropies
Data_CLAPS['Latenct'] = Latencies
import os
if not os.path.exists(File_name):
    os.mkdir(os.path.join('', File_name)) 

import json
dicts = json.dumps(Data_CLAPS)
with open(File_name + '/' + 'Data_CLAPS.json','w') as dicts_file:
    json.dump(dicts,dicts_file)


'''
