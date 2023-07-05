
"""
Exp_Basic_Diversification and clustering
"""

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



data = 'RIPE'

Dimention = 16*3
Goal = 'Diversification1'
DF  = Dataset(data,Dimention,Goal)


Data_set =  DF.data_set()


DF.plt_data()



Clustering = 'kmedoids'
Routing = True
Balancing = True
N = Dimention

num_Clusters = 5

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

rate = 100

num_targets = 20

Iterations = 1

run_time = 0.32

NYM = False
#
RIPE = True

strategy = 0

frac = 1

############################################Primary Expriments###################################################
##########################EXP1:Diversification+Routings##############################################################
# Routing is refered to all packet routing approaches: LowLatency(Imbalance) and Balanced (Greedy and Naive)
# Figures we dive in this experiment : Latency V.S. Tau(Both analytic and simulating approcehes), Entropy V.S. Tau(Both analytic and simulating approcehes), Entropy V.S. Latency (Just for Simulations)
Diversify = 0
C = Bridge(Data_set, Clustering, Diversify, Routing, Balancing, N, num_Clusters, speed_Function,Tau, r, Decimal_precision, Algorithm,Layers, mu,0,1,Goal,NYM,RIPE)


Iterations2 = 1
 



C.Entropy_Latency_VS_Tau(Iterations2)

del C
   
Sim = Simulation(num_targets,Iterations,Capacity,run_time,mu,Lambda,H_N,N,rate,Diversify,Clustering,0)       
        

Sim.Entropy_Latency_VS_Tau()

del Sim