 # -*- coding: utf-8 -*-
"""
Simulation :)
"""

        
    
    

# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle

from Message import message

from Mix_Node import Mix

from Mix_net import MixNet

from Message_Genartion_and_mix_net_processing import Message_Genartion_and_mix_net_processing

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

class Simulation(object):
    
    def __init__(self,Targets,Iteration,Capacity,run,delay1,delay2,H_N,N,rate,Diversify,T_C,ss,Goal ):
        self.Iterations = Iteration
        self.strategy = ss
        self.G = Goal
        self.CAP = Capacity
        self.rate = rate
        self.d1 = delay1
        self.d2 = delay2
        self.H_N = H_N
        self.N_target = Targets
        self.N = N
        self.run = run
        self.Diversify = Diversify
        if T_C != 'None':
            self.Parameter = 10
        else:
            self.Parameter = 0


    def compatible(self,A,B):
        a = len(A)
        b = len(B)
        X = []
    
        Min = min(a,b)
        if not a > Min:
            i = 0
            for item in A:
            
                X.append(B[i])
                i =i +1
            return A,X
        else:
            i = 0
            for item in B:
                X.append(A[i])
                i = i+1
            return X,B            
            
    

    def Simulator(self,corrupted_Mix,Latency,Mix_Dict): 
        import simpy
        Mixes = [] #All mix nodes
        env = simpy.Environment()    #simpy environment
        capacity=[]
        for j in range(self.N):# Generating capacities for mix nodes  
            c = simpy.Resource(env,capacity = self.CAP)
            capacity.append(c)           
        for i in range(self.N):#Generate enough instantiation of mix nodes  
            ll = i +1
            X = corrupted_Mix['PM%d' %ll]
            x = Mix(env,'M%02d' %i,capacity[i],X,self.N_target,self.d1)
            Mixes.append(x)

        MNet = MixNet(env,Latency,Mixes,self.N)  #Generate an instantiation of the mix net
        random.seed(42)  

        Process = Message_Genartion_and_mix_net_processing(env,Mixes,capacity,Mix_Dict,MNet,self.N_target,self.d2,self.H_N,self.rate)

        env.process(Process.Prc())  #process the simulation

        env.run(until = self.run)  #Running time

        
        Latencies = MNet.LL
       
        Latencies_T = MNet.LT
        Distributions = np.matrix(MNet.EN)
        DT = np.transpose(Distributions)
        ENT = []

        for i in range(self.N_target):
            llll = DT[i,:].tolist()[0]
            ENT.append(Ent(llll))
        return Latencies, Latencies_T,ENT




    def Entropy_Latency_VS_Tau(self):
        import numpy as np
        import json
        if self.Diversify ==0:
            File_name = 'Diversification_Basic_EXP' + self.G
        elif self.Diversify ==1:
            File_name = 'Random_Basic_EXP' + self.G
        elif self.Diversify == 2:
            File_name = 'WC_Basic_EXP' + self.G      
        
        with open(File_name + '/' +'Dict_save.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
         
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_tau in range(len(Dictionaries['Itr1'])):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['Itr%d'%Index_Itr]['Dic%d' %index]
                Mix_Dict = Dicts['Mix_Dict']

                        
                        
                    
                
                corrupted_Mix = Dicts['corrupted_Mix']
                
                
                Latency = Dicts['Latency'] 
               
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['Itr%d'%Index_Itr]['Dic%d' %index]
                Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']       
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['Itr%d'%Index_Itr]['Dic%d' %index]
                Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = []
        for Tau in np.arange(0,1.01,0.2):            
            T = round(100*Tau)/100
            labels.append(T)       
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'Tau':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/' +'Sim.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        





#########################################################################################
##############################################################################################
#########################Set 2 change the number of clusters#############################
##########################################################################################
###########################################################################################
        
        
    def Entropy_Latency_VS_Clusters(self,Tau):
        import json
        File_name = 'Num_Clusters'+ str(round(self.d1*1000))+'tau'+str(round(10*Tau))
        with open(File_name + '/'+ 'Dict_save_clusters.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_clusters in range(len(Dictionaries)):
            index = different_clusters + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Mix_Dict = Dicts['Mix_Dict']
                corrupted_Mix = Dicts['corrupted_Mix']
                Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']       
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)            
            
            
##################################################################################
##################################################################################
        labels = []
        NC_ =  [2,5,20,40,100,300]
        for Klusters in NC_:            
            K = round(100*Klusters)/100
            labels.append(K)
    
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'K':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/'+'Sim_change_clusters.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
        
    
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'Number of Clusters'
        Name = File_name + '/'+'Sim_Latency_VS_Clusters.png'
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = File_name + '/'+'Sim_Entropy_VS_Clusters.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)            
            
###########################################################################################################
###########################################################################################################           
###########################################################################################################
###########################################################################################################
#########################Set 3 change the number of mix nodes##############################################
###########################################################################################################
###########################################################################################################
                
    def Entropy_Latency_VS_Fraction_Of_N(self):
        import json
        File_name = 'Num_mix' + str(round(self.d1*1000)) + 'Diversify' + str(round(self.Diversify))
        with open(File_name + '/' + 'Dict_save_nodes.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        N = self.N
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_nodes in range(len(Dictionaries)):
            self.N = int((0.2+0.2*different_nodes)*N)
            self.W = int(self.N/3)
            index = different_nodes + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Mix_Dict = Dicts['Mix_Dict']
                corrupted_Mix = Dicts['corrupted_Mix']
                Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']       
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)            
            
            
##################################################################################
##################################################################################
        labels = []
        for I in range(5):        
            Ratio = 0.2+0.2*I            
            R = round(100*Ratio*self.N)/100
            labels.append(R)
    
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'N':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/' +'Sim_change_nodes.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'Mix nodes'
        Name = File_name + '/' + 'Sim_Latency_VS_nodes.png'
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = File_name + '/' + 'Sim_Entropy_VS_nodes.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)            
            
           
###########################################################################################################
###########################################################################################################           
###########################################################################################################
###########################################################################################################
#########################Set 4 change clustering method##############################################
###########################################################################################################
###########################################################################################################
  
    def Entropy_Latency_VS_Clustering_Method(self,Limitation):
        import json
        File_name = 'Clustering_Methods'
        with open(File_name + '/' +  'Dict_save_Methods.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_clusters in range(len(Dictionaries)):
            index = different_clusters + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Mix_Dict = Dicts['Mix_Dict']
                corrupted_Mix = Dicts['corrupted_Mix']
                Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']       
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)            
            
            
##################################################################################
##################################################################################
            if Limitation:
                Methods = ['kmeans','kmedoids']
            else:
                Methods = ['kmeans','kmedoids','FCM']  
###################################################################################            
#################################Saving the data###################################     
        df = {'Methods':Methods,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/' + 'Sim_change_Methods.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
        
    
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'Methods'
        Name = File_name + '/' + 'Sim_Latency_VS_Methods.png'
        from Plot import PLOT
        PLT = PLOT(Methods,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = File_name + '/' + 'Sim_Entropy_VS_Methods.png'
        PLT = PLOT(Methods,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)    
            
            
            
            
###########################################################################################################
###########################################################################################################           
###########################################################################################################
###########################################################################################################
#########################Advance EXP change r##############################################
###########################################################################################################
###########################################################################################################
  
    def Entropy_Latency_VS_Clustering_r(self):
        import json
        with open('Dict_save_r.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_clusters in range(len(Dictionaries)):
            index = different_clusters + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Mix_Dict = Dicts['Mix_Dict']
                corrupted_Mix = Dicts['corrupted_Mix']
                Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']       
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['NC1'])):
                Index_Itr = ITRS + 1
                Dicts = Dictionaries['NC%d'%index]['Dic%d' %Index_Itr]
                Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                Latency = Dicts['Latency']
                corrupted_Mix =Dicts['corrupted_Mix']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)            
            
            
##################################################################################
##################################################################################
        R = [ 0.01,0.05, 0.1,0.5, 1]
###################################################################################            
#################################Saving the data###################################     
        df = {'r':R,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open('Sim_change_r%d.json'%(self.Diversify+self.Parameter),'w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
        
    
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'r'
        Name = 'Sim_Latency_VS_r%d.png'%(self.Diversify + self.Parameter + 100*self.strategy)
        from Plot import PLOT
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = 'Sim_Entropy_VS_r%d.png'%(self.Diversify+self.Parameter+100*self.strategy)
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)    
 































           
##############################################################################################
##############################################################################################            
##################################Adversarial setting#########################################
##############################################################################################
##############################################################################################             
#######################################EXP1:change number of tau##############################
    def Adverserial_VS_Tau(self,Strategy):
        if self.Diversify ==0:
            File_name = 'Diversification_FCP'
        elif self.Diversify ==1:
            File_name = 'Random_FCP'
        elif self.Diversify == 2:
            File_name = 'WC_FCP'  

        import numpy as np
        import json
        with open(File_name + '/' + 'Dict_save_corrupted_tau.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []

        for different_tau in range(len(Dictionaries)):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = [] 

            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:

                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']  
                W = round(self.N/3)
                for i_ in range(2*W):
                    j_ = i_ + 1
                    SSS = Mix_Dict['PM%d'%j_]
                    factor = 0
                    for item_ in SSS:
                        factor = factor + item_
                    if factor>1:
                        factor_ = 0
                        for item__ in Mix_Dict['PM%d'%j_]:
                            Mix_Dict['PM%d'%j_][factor_] = item__/factor
                            factor_ = factor_ + 1

                        
                        
                        
                        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+1)]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                W = round(self.N/3)
                for i_ in range(2*W):
                    j_ = i_ + 1
                    SSS = Greedy_Balanced_Mix_Dict['PM%d'%j_]
                    factor = 0
                    for item_ in SSS:
                        factor = factor + item_
                    if factor>1:
                        factor_ = 0
                        for item__ in Greedy_Balanced_Mix_Dict['PM%d'%j_]:
                            Greedy_Balanced_Mix_Dict['PM%d'%j_][factor_] = item__/factor
                            factor_ = factor_ + 1                        
                        
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+2)]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                W = round(self.N/3)
                for i_ in range(2*W):
                    j_ = i_ + 1
                    SSS = Naive_Balanced_Mix_Dict['PM%d'%j_]
                    factor = 0
                    for item_ in SSS:
                        factor = factor + item_
                    if factor>1:
                        factor_ = 0
                        for item__ in Naive_Balanced_Mix_Dict['PM%d'%j_]:
                            Naive_Balanced_Mix_Dict['PM%d'%j_][factor_] = item__/factor
                            factor_ = factor_ + 1                         
                        
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = []
        for Tau in np.arange(0,1.01,0.2):            
            T = round(100*Tau)/100
            labels.append(T)       
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'Tau':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/' + str(Strategy) + 'Sim_Corrupted_tau.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        


#################################################################################################
#################################################################################################
#################################################################################################
##############################################################################################
##############################################################################################             
#######################################EXP2:change number of C_Nodes##############################
    def Adverserial_VS_CNodes(self,Strategy):
        import numpy as np
        import json
        if self.Diversify ==0:
            File_name = 'Diversification_FCP_CNodes'
        elif self.Diversify ==1:
            File_name = 'Random_FCP_CNodes'
        elif self.Diversify == 2:
            File_name = 'WC_FCP_CNodes'      
                
        
        
        with open(File_name + '/' + 'Dict_save_corrupted_Cnodes.json','r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_tau in range(len(Dictionaries)):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+1)]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+2)]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = []
        for CN in [0.05,0.1,0.15,0.2]:            
            T = round(100*CN)/100
            labels.append(T)       
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'CN':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open(File_name + '/' + str(Strategy) + 'Sim_Corrupted_CNodes.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        

#################################################################################################
#################################################################################################
#################################################################################################
##############################################################################################
##############################################################################################             
#######################################EXP2:change number of clusters##############################
    def Adverserial_VS_C_Clusters(self,Strategy):
        import numpy as np
        import json
        with open('Dict_save_corrupted_clusters%d.json'%self.Diversify,'r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_tau in range(len(Dictionaries)):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+1)]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+2)]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = []
        for K_ in [round(self.N/5),round(2*self.N/5),round(3*self.N/5),round(4*self.N/5)]:            
            T = round(100*K_)/100
            labels.append(T)       
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'K':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open('Sim_Corrupted_C_Clusters%d.json'%((self.Diversify+self.Parameter)*Strategy),'w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'Number of clusters'
        Name = 'Sim_Latency_Corrupted_C_Clusters%d.eps'%((self.Diversify + self.Parameter + 100*self.strategy)*Strategy)
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = 'Sim_Entropy_Corrupted_C_Clusters%d.eps'%((self.Diversify+self.Parameter+100*self.strategy)*Strategy)
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)

#################################################################################################
#################################################################################################
#################################################################################################
##############################################################################################             
#######################################EXP2:change clustering##############################
    def Adverserial_VS_C_Methods(self,Strategy,Limitation):
        if Limitation:
            Methods = ['kmeans','kmedoids']
        else: 
            Methods = ['kmeans','kmedoids','FCM']
        import numpy as np
        import json
        with open('Dict_save_corrupted_Methods%d.json'%self.Diversify,'r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_tau in range(len(Dictionaries)):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+1)]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+2)]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = Methods 
###################################################################################            
#################################Saving the data###################################     
        df = {'Methods':Methods,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open('Sim_Corrupted_C_Methods%d.json'%((self.Diversify+self.Parameter)*Strategy),'w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'Clustering'
        Name = 'Sim_Latency_Corrupted_C_Methods%d.eps'%((self.Diversify + self.Parameter + 100*self.strategy)*Strategy)
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = 'Sim_Entropy_Corrupted_C_Methods%d.eps'%((self.Diversify+self.Parameter+100*self.strategy)*Strategy)
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
#################################################################################################
##############################################################################################
##############################################################################################             
#######################################EXP2:change r##############################
    def Adverserial_VS_C_r(self,Strategy):
        import numpy as np
        import json
        with open('Dict_save_corrupted_r%d.json'%self.Diversify,'r') as dicts:
            Dictionaries = json.loads(json.load(dicts))
        Latency_tau_Imbalance = []
        Latency_tau_Imbalance_T = []    
        Entropy_tau_Imbalance = []
        Latency_tau_Greedy_Balance = []        
        Latency_tau_Greedy_Balance_T = []    
        Entropy_tau_Greedy_Balance = []
        Latency_tau_Naive_Balance = []        
        Latency_tau_Naive_Balance_T = []    
        Entropy_tau_Naive_Balance = []
        for different_tau in range(len(Dictionaries)):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries['DicT1'])):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Mix_Dict = Dicts['Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']    
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Imbalance.append(End_to_End_Latancy_Vector)
            Latency_tau_Imbalance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Imbalance.append(Message_Entropy_Vector)
########################################Greedy Balanced###########################################
##########################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+1)]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Greedy_Balanced_Mix_Dict = Dicts['Greedy_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency,Greedy_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Greedy_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Greedy_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Greedy_Balance.append(Message_Entropy_Vector)                      
#####################################Naive Blanced#############################################
##################################################################################
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            for ITRS in range(len(Dictionaries)):
                Index_Itr = ITRS + 1
                if Strategy == 6:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%(Strategy+2)]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']                     
                else:
                    Dicts = Dictionaries['DicT%d'%index]['Dic%d' %Index_Itr]['Strategy%d'%Strategy]
                    Naive_Balanced_Mix_Dict = Dicts['Naive_Balanced_Mix_Dict']
                    if Strategy ==5:
                        corrupted_Mix = Dicts['corrupted_Mix'][0]
                    else:
                        corrupted_Mix = Dicts['corrupted_Mix']
                    Latency = Dicts['Latency']        
                for Iteration in range(self.Iterations):
                    Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Latency, Naive_Balanced_Mix_Dict)
                    End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                    End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                    Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_tau_Naive_Balance.append(End_to_End_Latancy_Vector)
            Latency_tau_Naive_Balance_T.append(End_to_End_Latancy_Vector_T)
            Entropy_tau_Naive_Balance.append(Message_Entropy_Vector)
##################################################################################
##################################################################################
        labels = []
        for rr in [0.001, 0.01, 0.1, 1, 10, 100]:            
            T = round(100*rr)/100
            labels.append(T)       
        for i in range(len(labels)):
            labels[i] = int(labels[i]*100)/100   
###################################################################################            
#################################Saving the data###################################     
        df = {'r':labels,
            ' Latancy(Im)_T': Latency_tau_Imbalance_T,
            'Latency(Im)' : Latency_tau_Imbalance,
            'Entropy(Im)' : Entropy_tau_Imbalance,
            ' Latancy(Gr)_T': Latency_tau_Greedy_Balance_T,
            'Latency(Gr)' : Latency_tau_Greedy_Balance,
            'Entropy(Gr)' : Entropy_tau_Greedy_Balance,            
            ' Latancy(Na)_T': Latency_tau_Naive_Balance_T,
            'Latency(Na)' : Latency_tau_Naive_Balance,
            'Entropy(Na)' : Entropy_tau_Naive_Balance             
                              }
        import json
        dics = json.dumps(df)
        with open('Sim_Corrupted_C_r%d.json'%((self.Diversify+self.Parameter)*Strategy),'w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
##################################Plots##################################################           
        D = ['Low Latency(Imbalance)','Greedy Balance','Naive Balance']
        Y = [Latency_tau_Imbalance ,Latency_tau_Greedy_Balance, Latency_tau_Naive_Balance ]
        Y_Label = 'Latency(sec)'
        X_Label = 'r'
        Name = 'Sim_Latency_Corrupted_C_r%d.eps'%((self.Diversify + self.Parameter + 100*self.strategy)*Strategy)
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        
        
        Y = [Entropy_tau_Imbalance ,Entropy_tau_Greedy_Balance, Entropy_tau_Naive_Balance ]
        Y_Label = 'Entropy(bit)'
        Name = 'Sim_Entropy_Corrupted_C_r%d.eps'%((self.Diversify+self.Parameter+100*self.strategy)*Strategy)
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)

#################################################################################################
#################################################################################################
#################################################################################################        
             
              
        
                        
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
      
            


        
        