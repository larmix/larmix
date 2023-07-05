# -*- coding: utf-8 -*-
"""
Topology to Distance to Latency: In this class we aim to measure 
distances and latencies between diffrent mix nodes
"""
import numpy as np

from math import exp
from scipy import constants


def I_key_finder(x,y,z,G,NYM = False):
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
    


class Latency_and_Distance(object):
    
    def __init__(self,Topology,Tranformation_Type,Goal,NYM = False,RIPE = True ):
        import numpy as np
        self.Mix_net = Topology
        (a,b,c) = np.shape(self.Mix_net)
        self.W = b
        self.L =a
        self.N =round(a*b)
        self.speed_func = Tranformation_Type #Means the method you desire for 
        #transfering mix node distances to latency
        self.NYM = NYM
        self.RIPE = RIPE
        self.G = Goal

        
        
        
        
    def Topology_to_Distance(self):#Try to receive the mix nodes  location and 

        if True:

            RIPE_Latency = []
            import json

            with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 

                data0 = json.load(json_file)            
            
            Dis=np.zeros((self.W,self.W,self.L-1))
            for j in range(self.L-1):
                for i in range(self.W):
                    x = self.Mix_net[j,i,0]
                    y = self.Mix_net[j,i,1]                    
                    z = self.Mix_net[j,i,2]                                        
                    I_key = I_key_finder(x, y, z,self.G,False)
                    Index = Loc_finder(I_key,data0)
                    LIST_Latency = []
                    for k in range(self.W):
                        x = self.Mix_net[j+1,k,0]
                        y = self.Mix_net[j+1,k,1]                    
                        z = self.Mix_net[j+1,k,2]                         
                        I_key2 = I_key_finder(x, y, z,self.G,False)
                        In_Latency = data0[Index]['latency_measurements'][str(I_key2)]
  
                        delay_distance = int(In_Latency)
                        if delay_distance == 0:
                            delay_distance =1
                        Dis[i,k,j] = delay_distance
                        LIST_Latency.append(delay_distance/2000)
                    RIPE_Latency.append(LIST_Latency)

            self.RIPE_Latency = RIPE_Latency            
 
               
        return Dis 
    
    
    

    


    
    def Distance_to_time(self,x):#Function for transferring distances to latency

        if self.speed_func == '2/3c':
            speed = ((2/3) * constants.c) / 1000
            return x / speed

        elif self.speed_func == '1/3c':
            speed = ((1/3) * constants.c) / 1000
            return x / speed

        elif self.speed_func == 'Verloc':
            # ! The empirical speed function uses meters as input
            if x>5900:
                ratio = int(x/600)
                x = 5900
                return (x*100*ratio)/(5.817e+07 * exp(1.645e-07*(x*1000)) -4.785e+07 * exp(-2.812e-06*(x*1000)))            
            return (x*1000)/(5.817e+07 * exp(1.645e-07*(x*1000)) -4.785e+07 * exp(-2.812e-06*(x*1000))) 
        
    def get_time(self, x):

        try:
            time = self.Distance_to_time(x)
        except Exception as e:
            print ('Failed in Propagation time_func:', e, 'x=', x)
            time = 0
        return time
    
    
    def Distance_to_Latency(self):

        Latency = []
        Distances  = self.Topology_to_Distance()
        for i in range (self.L-1):
            for j in range(self.W):
                L = []
                for k in range(self.W):
                    
                    L.append(self.get_time(Distances[j,k,i]))
                Latency.append(L)
        return Latency
                    
                    
                    


        
    def Distance_to_real_end_to_end_Latency(self,Matrix):

        (a,b,c) = np.shape(Matrix)
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    Matrix[i,j,k] = self.Distance_to_time(Matrix[i,j,k])
                    return Matrix
        
        
   
 












