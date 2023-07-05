# -*- coding: utf-8 -*-
"""
Client Latency
"""
import numpy as np

from math import exp
from scipy import constants
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


def Ent_T(Policy,Type,W):
    a = 0
    for i in range(W):
        a = a + Ent(Policy['PM'+str(i+1)+Type])
    return a/W



def I_key_finder(x,y,z,G,NYM = True):
    import json
    
    if NYM:

        with open('NYM.json','r') as dicts:
            data = json.loads(json.load(dicts))
    else:
        with open(G + 'RIPE.json','r') as dicts:
            data = json.loads(json.load(dicts))        
    
    for i in range(len(data['x'])):
        if int(x*1000) == int(1000*data['x'][i]) and int(y*1000) == int(1000*data['y'][i]) and int(z*1000) == int(1000*data['z'][i]):
            I_key = data['i_key'][i]
    
    return I_key
            

def Loc_finder(I_key,data):
    for i in range(len(data)):
        if data[i]['i_key'] == I_key:
            return i
def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

class Client_Distance_Network(object):
    
    def __init__(self,Frac,N,Goal):
        self.Frac = Frac
        self.N = N
        self.Goal = Goal
        self.n1 = int(self.Frac*self.N)
        self.n2 = int((1-self.Frac)*self.N)
        

    def Data(self):
        from Datasets_ import Dataset
    
        D = Dataset('RIPE',self.n1,self.Goal,self.n2,1)
    
        D.RIPE()
        a,b,c = D.PLOT_New_dataset()
        return a,b
    
    def Distance(self,D1,D2):#Try to receive the mix nodes  location and 
    
        import json
    
        with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 
    
            data0 = json.load(json_file)            
                
    
                                           
        I_key = I_key_finder(D1[0], D1[1], D1[2],self.Goal,False)
        Index = Loc_finder(I_key,data0)
                                         
        I_key2 = I_key_finder(D2[0], D2[1], D2[2],self.Goal,False)
        In_Latency = data0[Index]['latency_measurements'][str(I_key2)]
      
        delay_distance = int(In_Latency)
        if delay_distance == 0:
            delay_distance =1
        return delay_distance/2000
    
    def Ave_Distance(self,Iteration):
        import numpy as np
        Latency = []
        for i in range(Iteration):
            Latency_inner = []
        
            Client_data,MixNodes = self.Data() 
            for j in range(self.n1):
                for k in range(self.n2):
                    Latency_inner.append(self.Distance(Client_data[j,:].tolist()[0],MixNodes[k,:].tolist()[0]))
            Latency.append(sum(Latency_inner)/len(Latency_inner))
        return sum(Latency)/len(Latency)
            
            
            
            
            
            
     














