# -*- coding: utf-8 -*-
"""
2Layers
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





def DataSet(Data_Type,Goal,N):#Try to receive the mix nodes  location and 
    from Loopix_Data import Dataset

    Data = Dataset(Data_Type,N,Goal)

    Mix_Data = Data.data_set()
    W = int(N/2)
        
    RIPE_Latency = {}
    import json

    with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 

        data0 = json.load(json_file)            
            
        for i in range(W):
            x = Mix_Data[i,0]
            y = Mix_Data[i,1]
            z = Mix_Data[i,2]
                                       
            I_key = I_key_finder(x, y, z,Goal,False)
            Index = Loc_finder(I_key,data0)
            for j in range(W):                
                x = Mix_Data[j+W,0]
                y = Mix_Data[j+W,1]
                z = Mix_Data[j+W,2]                         
                I_key2 = I_key_finder(x, y, z,Goal,False)
                In_Latency = data0[Index]['latency_measurements'][str(I_key2)]
  
                delay_distance = int(In_Latency)
                if delay_distance == 0:
                    delay_distance =1
                RIPE_Latency['PM'+str(i+1)+'PM'+str(W+j+1)] = delay_distance/2000
 
    return RIPE_Latency

def Loopix_two_layers(N,Dataset,Goal,Iteration,Mix_delay,Poisson_delay,running,Num_Target,tau):
    if tau == 0:
        tau = 0.01
    L_A_Im = []
    L_A_Gr = []  
    H_A_Im = []
    H_A_Gr = []    
    L_S_Gr = []
    H_S_Gr = []
    L_S_Im = []
    H_S_Im = []    
    for i in range(Iteration):       
        L = DataSet('RIPE','Loopix'+str(N),N)
        C = Loopix(int(N/2),L,Mix_delay,Num_Target,running,Poisson_delay,tau)
        
        L_A_Im.append(C.AV_Latency_Im)
        L_A_Gr.append(C.AV_Latency_Gr)   
        H_A_Im.append(C.Analytic_H_Im)
        H_A_Gr.append(C.Analytic_H_Gr)  
        C.Simulation('Gr')
        L_S_Gr= L_S_Gr + C.Sim_Latency
        H_S_Gr= H_S_Gr + C.Sim_Entropy

        C.Simulation('Im')
        L_S_Im= L_S_Im + C.Sim_Latency
        H_S_Im= H_S_Im + C.Sim_Entropy
        
    h_s_gr = Med([H_S_Gr])
    l_s_gr = Med([L_S_Gr])
    h_s_im = Med([H_S_Im])
    l_s_im = Med([L_S_Im])    
    l_a_gr = Med([L_A_Gr])
    l_a_im = Med([L_A_Im]) 
    h_a_gr = Med([H_A_Gr])
    h_a_im = Med([H_A_Im])     
    
    Dict = {'H_A_Gr':h_a_gr,'H_A_Im':h_a_im,'H_S_Gr':h_s_gr,'H_S_Im':h_s_im,'L_A_Gr':l_a_gr,'L_A_Im':l_a_im,'L_S_Gr':l_s_gr,'L_S_Im':l_s_im}
    import json
    File_name = Goal
    dicts = json.dumps(Dict)
    import os
         
    if not os.path.exists(File_name):
        os.mkdir(os.path.join('', File_name))     
    with open(File_name + '/' + 'Dict_save.json','w') as dicts_file:
            json.dump(dicts,dicts_file)
    return     Dict 

class Loopix(object):
    
    def __init__(self,W,Latency,mix_delay,Target,run,P_delay,tau):
        import numpy as np
        import math
        self.tau = tau
        from Routing_ import Routing
        R = Routing(self.tau,0,0,0,0,1)
        Policy = {}
        
        self.W = W
        self.l = 2
        self.N = self.W*self.l
        self.Latency = Latency
        self.L = []
        Imbalanced = []
        for i in range(self.W):
            List = []
            for j in range(self.W):
                List.append(self.Latency['PM'+str(i+1)+'PM'+str(self.W+j+1)])
            Policy['PM'+str(i+1)+'Im'] = R.PDF(List)
            Imbalanced.append(Policy['PM'+str(i+1)+'Im'])

        import numpy as np
        Im_Matrix = np.matrix(Imbalanced)

        from Balnced_ import Balanced_Layers
        Balance = Balanced_Layers(Im_Matrix,5,'Greedy')
        Balance.make_the_layer_balanced()
        Gr_Matrix = Balance.IMD

        for i in range(self.W):
            Policy['PM'+str(i+1)+'Gr'] = Gr_Matrix[i,:].tolist()[0]
        self.R_Policy = Policy

            




        Av_Gr_Latency = []
        Av_Im_Latency = []
        for i in range(self.W):
            f1 = self.R_Policy['PM' + str(i+1)+'Im']
            f2 = self.R_Policy['PM' + str(i+1)+'Gr']
            
            for j in range(self.W):
                Av_Gr_Latency.append((1/self.W)*f2[j]*self.Latency['PM'+str(i+1)+'PM'+str(self.W+j+1)])
                Av_Im_Latency.append((1/self.W)*f1[j]*self.Latency['PM'+str(i+1)+'PM'+str(self.W+j+1)])
                

        self.AV_Latency_Im = sum(Av_Im_Latency)
            
        self.AV_Latency_Gr = sum(Av_Gr_Latency)

        self.Analytic_H_Gr = Ent_T(self.R_Policy,'Gr',self.W)
        self.Analytic_H_Im = Ent_T(self.R_Policy,'Im',self.W)        
        self.mix_delay = mix_delay
        self.N_T = Target
        self.run = run
        self.delay_Poisson = P_delay
        
        self.Corrupted_Mixes = {}
        for k in range(self.N):
            self.Corrupted_Mixes['PM'+str(k+1)] = False
        
                        


        
        
        
    def Simulation(self,Type):
        from Sim import Simulation
        
        S = Simulation(self.N_T,self.run,self.mix_delay,self.delay_Poisson,self.N)
        
        self.Sim_Latency,self.Sim_Entropy = S.Simulator(self.Corrupted_Mixes,self.Latency,self.R_Policy,Type)
        
        
        
'''
#Test Loopix Class:
W = 3
mix_delay = 0.05
Target = 20
run = 0.7
P_delay = 0.0001
tau = 0.6
Latency = {'PM1PM4': 0.12, 'PM1PM5': 0.12, 'PM1PM6': 0.03, 'PM2PM4': 0.11, 'PM2PM5': 0.113, 'PM2PM6': 0.115, 'PM3PM4': 0.04, 'PM3PM5': 0.02, 'PM3PM6': 0.07}
        
print(Latency)
c = Loopix(W,Latency,mix_delay,Target,run,P_delay,tau)

c.Simulation('Gr')

print(c.Sim_Latency,c.Sim_Entropy)

'''

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




































