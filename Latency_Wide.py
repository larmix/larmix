# -*- coding: utf-8 -*-
"""
Latency for new setting

"""
import numpy as np

from math import exp
from scipy import constants


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
    


class Latency_and_Dict(object):
    
    def __init__(self,Topology,Tranformation_Type,Goal,Client_Data,GW_Data,Scenario = False, NYM = False,RIPE = False ):
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
        self.client_Data = Client_Data
        self.GW_Data = GW_Data
        self.Scenario = Scenario
        


        
        
        
        
    def Topology_to_Distance(self):#Try to receive the mix nodes  location and 

        
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


    def GW_Mixes_Client(self):

        
        GW_delay = {}
        gg = 'G'
        mm = 'PM'
        cc = 'Cl'
        import json

        with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 

            data0 = json.load(json_file)    
        
        G = 0
        for Row in self.GW_Data:
            G = G +1
            
            row = Row.tolist()[0]
            x = row[0]
            y = row[1]                    
            z = row[2]                                       
            I_key = I_key_finder(x, y, z,self.G,False)
            Index = Loc_finder(I_key,data0)
            for j in range(self.L):
                for k in range(self.W):                    
                    x = self.Mix_net[j,k,0]
                    y = self.Mix_net[j,k,1]                    
                    z = self.Mix_net[j,k,2]                         
                    I_key2 = I_key_finder(x, y, z,self.G,False)
                    In_Latency = data0[Index]['latency_measurements'][str(I_key2)]
  
                    delay_distance = int(In_Latency)
                    if delay_distance == 0:
                        delay_distance =1
                    
                    
                    GW_delay[mm+str(j*self.W+k+1)+gg+str(G)] = delay_distance/2000
            CC = 0
            for Row_ in self.client_Data:
                CC = CC + 1
                row_ = Row_.tolist()[0]
                x = row_[0]
                y = row_[1]                    
                z = row_[2] 
                I_key2 = I_key_finder(x, y, z,self.G,False)
                In_Latency = data0[Index]['latency_measurements'][str(I_key2)]
  
                delay_distance = int(In_Latency)
                if delay_distance == 0:
                    delay_distance =1 
                GW_delay[cc+str(CC)+gg+str(G)] = delay_distance/2000
                
        if self.Scenario:
            
            GW_Dict = {}
            for i in range(self.N):
                j = i+1
                M = 1000
                for g in range(len(self.GW_Data)):
                    g1 = g + 1
                    if GW_delay[mm+str(j)+gg+str(g1)]<M:
                        M = GW_delay[mm+str(j)+gg+str(g1)]
                        GW_Dict[mm+str(j)] = gg+str(g1)
        
            for i in range(len(self.client_Data)):
                j = i+1
                Cl = 1000
                for g in range(len(self.GW_Data)):
                    g1 = g + 1
                    if GW_delay[cc+str(j)+gg+str(g1)]<Cl:
                        Cl = GW_delay[cc+str(j)+gg+str(g1)]
                        GW_Dict[cc+str(j)] = gg+str(g1)                    
        else:
            
            GW_Dict = {}
            for g in range(len(self.GW_Data)):
                
                A = []
                g1 = g + 1
                for i in range(self.W):
                    j = i+1
                    

                    A.append(GW_delay[mm+str(j)+gg+str(g1)])
                GW_Dict[gg+str(g1)] = A
        
            for i in range(len(self.client_Data)):
                j = i+1
                B = []
                for g in range(len(self.GW_Data)):
                    g1 = g + 1
                    B.append(GW_delay[cc+str(j)+gg+str(g1)])
                GW_Dict[cc+str(j)] = B               
                
                
                
               
        return GW_delay , GW_Dict
    
    

    
    
    

    
from Datasets import Dataset

 
import numpy as np
'''
n = 3
cc = 5
Topology = np.zeros((3,n,3))




D = Dataset('RIPE',3*n,'rr',cc,n)

D.RIPE()
a,b,c = D.PLOT_New_dataset()





Topology[0,:,:] = a[0:n,:]

Topology[1,:,:] = a[n:2*n,:]
Topology[2,:,:] = a[2*n:3*n,:]



L = Latency_and_Dict(Topology, 'a', 'rr', b, c)


A = L.Topology_to_Distance()


B,C = L.GW_Mixes_Client()

print(B)

'''
