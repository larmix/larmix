
"""
Maximum mixing delays
"""

def Path_Fraction(a,b,c,Dict,W):
    Term = 0
    if len(a) !=0 and len(b) !=0 and len(c) !=0:
        for item1 in a:
            for item2 in b:
                for item3 in c:
                    Term = Term + (1/W)*(Dict['PM%d' %item1][item2-W-1])*(Dict['PM%d' %item2][item3-2*W-1])
    return Term

def Accuracy(List,accuracy):
    Cases = 0
    for item in List:
        if not item > accuracy:
            Cases = Cases + 1
    return Cases/len(List)

def nCr(n,r):
    import math
    f = math.factorial
    return f(n) // f(r) // f(n-r)
def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


def sort_of_clusters(Labels1):
    lists = Labels1
    Index1 = []
    for i in range(len(lists)):
        maxs = 0
        j = 0
        for item in lists:
            if item > maxs:
                maxs = item
                index1 = j
            j = j +1
        lists[index1] = 0
        Index1.append(index1)
    return Index1

import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def sort_index(List , X):
    x = 0
    INDEX = []
    while(x < X):
        Max = max(List)
        Indx = List.index(Max)
        INDEX.append(Indx)
        List[Indx] = 0
        x = x+1
    return INDEX
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

class Detailed_Analysis(object):

    import math
    import numpy as np

    def __init__(self,Data_set,Clustering,Diversify,Routing,Balancing,N,num_Clusters,
                 Speed_Function,tau,H_parameter,Decimal_A,Algorithm,Layers,mu,strategy,Frc,Targets,Iteration,Capacity,run,delay2,H_N,rate,T_C,Goal,NYM = False,RIPE = False):
        self.NYM = NYM
        self.RIPE = RIPE
        self.G = Goal
        self.strategy = strategy
        self.frc = Frc
        self.data = Data_set
        
        self.Clustering = Clustering
        
        if self.Clustering != 'None':
            self.Parameter = 10
        else:
            self.Parameter = 0
        
        self.Diversify = Diversify
        
        self.Routing = Routing
        
        self.Balancing = Balancing
        
        self.N = N
        self.K = num_Clusters
        if self.Diversify ==2:
            self.K = 3
        
        self.Speed_Function = Speed_Function
        
        self.Distance , self.Latency = self.mix_net_()

        self.tau = tau
        
        self.H_parameter = H_parameter
        
        self.DC = Decimal_A
        
        
        self.Algorithm = Algorithm
        
        self.L = Layers
        self.W = round(self.N/self.L)
        
        self.I_Distributions , self.Gready_Balanced_Distributions, self.Naive_Balanced_Distributions = self.Rout_and_Balance()

        self.mu = mu
        self.Corruption = 0
        
        self.OneTime = True

        self.Iterations = Iteration
        self.strategy = strategy
        self.CAP = Capacity
        self.rate = rate
        self.d1 = mu
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

    def mix_net_(self):
        
        

        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance

        
        from Corruption import corruptedMix
        if self.strategy == 2:
            import numpy as np
            MM = np.zeros((self.N,3))
            LM = 0
            
            Class_CNs = corruptedMix(self.data,0.3,MM,LM,self.strategy)
            Class_CNs.corrupted_mix_nodes()
            self.Corruption = Class_CNs.CNs
            clusters = Clustering(self.data,self.Clustering,self.K,3,self.Corruption)
            C_C = clusters.mapping()

            clusters.Data_plt()
        
            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C_C,self.frc,True)
        
            arrangment.Topology_plt()
            
            self.Corruption_New = arrangment.mapping()
        
            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)
            
            if (not self.NYM) and  (not self.RIPE):
                Distances_ = LatencyDistance.Topology_to_Distance()
                Distances = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_)
                Latency = LatencyDistance.Distance_to_Latency()
            else:
                if self.NYM:
                    Distances = LatencyDistance.Topology_to_Distance()
                    Latency = LatencyDistance.NYM_Latency
                elif self.RIPE:
                    Distances = LatencyDistance.Topology_to_Distance()
                    Latency = LatencyDistance.RIPE_Latency                
            

             
        else:
            
            if self.strategy == 6:
                clusters = Clustering(self.data,self.Clustering,self.K,3,0)
                arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,0,self.frc,True)
        
                arrangment.Topology_plt()
            

        
                LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)

                if (not self.NYM) and  (not self.RIPE):
                    Distances_ = LatencyDistance.Topology_to_Distance()
                    Distances = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_)
                    Latency = LatencyDistance.Distance_to_Latency()
                else:
                    if self.NYM:
                        Distances = LatencyDistance.Topology_to_Distance()
                        Latency = LatencyDistance.NYM_Latency
                    elif self.RIPE:
                        Distances = LatencyDistance.Topology_to_Distance()
                        Latency = LatencyDistance.RIPE_Latency                
                
            else:
                import numpy as np


                clusters = Clustering(self.data,self.Clustering,self.K,3,0)

                clusters.Data_plt()
                Class_CNs = corruptedMix(self.data,0.3,clusters.Mixes,clusters.Labels,self.strategy)
                Class_CNs.corrupted_mix_nodes()
                self.Corruption = Class_CNs.CNs        
                arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,self.Corruption,self.frc,True)
        
                arrangment.Topology_plt()
            
                self.Corruption_New = arrangment.mapping()
        
                LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)
                if (not self.NYM) and  (not self.RIPE):
    
                    Distances_ = LatencyDistance.Topology_to_Distance()
                    Distances = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_)
                    Latency = LatencyDistance.Distance_to_Latency()
                else:
                    if self.NYM:

                        Distances = LatencyDistance.Topology_to_Distance()
                        Latency = LatencyDistance.NYM_Latency
                    elif self.RIPE:
                        Distances = LatencyDistance.Topology_to_Distance()
                        Latency = LatencyDistance.RIPE_Latency
                


        return Distances, Latency
        
        
        

    def Rout_and_Balance(self):
        import numpy as np
        
        from Routing import Routing
        from Balnced import Balanced_Layers
        


        R = Routing(self.tau,self.Distance,self.Routing,self.H_parameter)
        
        Initial_Dist = R.Rout()

        (p,q,Z) = np.shape(Initial_Dist)
        I_D = []
        for k in range(Z):
            for I in range(p):
                for J in range(q):
                    I_D.append(Initial_Dist[I,J,k])
        


        (a,b,c) = np.shape(Initial_Dist)
        
        Balanced_Dist = np.zeros((a,b,c))
        
        Naive_Balanced = np.zeros((a,b,c))
        
        if self.Routing:
        
            Balance1 = Balanced_Layers(Initial_Dist[:,:,0],self.DC,self.Algorithm)
        
            Balance2 = Balanced_Layers(Initial_Dist[:,:,1],self.DC,self.Algorithm)
            
            Balance1.make_the_layer_balanced()
            
            Balance2.make_the_layer_balanced()
        
        
            Balanced_Dist[:,:,0] = Balance1.IMD
        
            Balanced_Dist[:,:,1] = Balance2.IMD

        
            Naive_Balanced[:,:,0] = Balance1.Naive
        
            Naive_Balanced[:,:,1] = Balance2.Naive
            

        else:
            
            Balanced_Dist = Initial_Dist
            
            Naive_Balanced = Initial_Dist
        II = np.zeros((p,q,Z))
        Counter = 0
        for I in range(c):
            for J in range(a):
                for K in range(b):
                    II[J,K,I] = I_D[Counter]
                    Counter = Counter + 1

        return II, Balanced_Dist, Naive_Balanced

    def Make_the_Analysis(self):
        from Prior_Analysis import Prior_Analysis
        mu = self.mu
        Analysis = Prior_Analysis(self.I_Distributions , self.Gready_Balanced_Distributions,
                                   self.Naive_Balanced_Distributions,self.Latency)
        Entropy_of_TransformationMatrix_Imbalanced = Analysis.Entropy(0)
        Entropy_of_TransformationMatrix_Balanced_G = Analysis.Entropy(1)
        Entropy_of_TransformationMatrix_Balanced_N = Analysis.Entropy(2)
        
        Average_Latency1 = Analysis.Ave_Latency(0,mu)
        Average_Latency2 = Analysis.Ave_Latency(1,mu)
        Average_Latency3 = Analysis.Ave_Latency(2,mu)
        
        Entropy =[Entropy_of_TransformationMatrix_Imbalanced,Entropy_of_TransformationMatrix_Balanced_G,Entropy_of_TransformationMatrix_Balanced_N]
        Latency = [Average_Latency1,Average_Latency2,Average_Latency3]
        return Entropy,Latency


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

    def Entropy_Latency_Analytic(self,Iterations,Bound):
        File_name = 'Maximum_Mixing_Delays' +str(round(1000*Bound))
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))          
        import numpy as np  
        
        self.d1 = 0
        corrupted_Mix = self.Corruption_New #Assign the corrupted mix nodes to the corrupted dic.
        for k in range(1,self.N+1):
            corrupted_Mix['PM%d' %k] = False        
        TAU = []        
        L_VS_Tau = []
        E_VS_Tau = []
        Dictionaries = {}
        CNTR = 0  
        for Realization in range(Iterations):
            CNTR = CNTR + 1
            D_ , L_ = self.mix_net_()        
            self.Distance = D_
            self.Latency = L_
            Latency_List = []
            Entropy_List = []
             
            Dicts = {}
            current_dic = 1             
            for Tau in np.arange(0,1.01,0.1):
                if Tau==0:
                    Tau = 0.01
                self.tau = Tau
                if Realization == 0:
                    TAU.append(Tau)

                AA,BB,CC= self.Rout_and_Balance()

            
                self.I_Distributions = AA
            
                self.Gready_Balanced_Distributions = BB
            
                self.Naive_Balanced_Distributions = CC
                
                Mix_Dict ={}
                Balanced_Greddy_Mixes = {}
                Balanced_Naive_Mixes = {}
        
 
                counter = 0
                for j in range( self.L-1):
                    for i in range(self.W):
                        counter = counter + 1
                        Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                        Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                        Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
       
                Dict={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
               'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': corrupted_Mix,'Latency': self.Latency}
                Dicts['Dic%d' %current_dic] = Dict
                current_dic = current_dic + 1
            

                H,L = self.Make_the_Analysis()
                Latency_List.append(L)
                Entropy_List.append(H)



            LL1 = np.transpose(np.matrix(Latency_List))
            LL2 = LL1.tolist()

            HH1 = np.transpose(np.matrix(Entropy_List))
            HH2 = HH1.tolist()
            L_VS_Tau.append(LL2)
            E_VS_Tau.append(HH2)
            Dictionaries ['Itr%d'%CNTR] = Dicts

        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'M_Detailed.json','w') as dicts_file:
            json.dump(dicts,dicts_file)

        for TERM in range(3):
            for ITEM in range(len(L_VS_Tau)):
                List1 = []
                List2 = []
                List1.append(L_VS_Tau[ITEM][TERM])
                List2.append(E_VS_Tau[ITEM][TERM])            
            LLL1 = np.transpose(np.matrix(List1))
            LLL2 = LLL1.tolist()
            HHH1 = np.transpose(np.matrix(List2))
            HHH2 = HHH1.tolist()            
            if TERM == 0:
                H_Imbalanced = Med(HHH2)
                L_Imbalanced = Med(LLL2)
            elif TERM == 1:
                H_Greedy =   Med(HHH2)
                L_Greedy =   Med(LLL2)                
            elif TERM == 2:

                H_Naive =  Med(HHH2)
                L_Naive =  Med(LLL2) 
            
                        


        import pandas as pd
        Max_mixing_I = []
        Max_mixing_G = []
        Max_mixing_N = []        
        for i in range(len(TAU)):
            TAU[i] = int(TAU[i]*100)/100
            L_Imbalanced[i] = int(1000*L_Imbalanced[i])/1000
            Max_mixing_I.append((Bound-L_Imbalanced[i])/3)
            H_Imbalanced[i] = int(1000*H_Imbalanced[i])/1000
            
            L_Greedy[i] = int(1000*L_Greedy[i])/1000
            Max_mixing_G.append((Bound-L_Greedy[i])/3)            
            H_Greedy[i] = int(1000*H_Greedy[i])/1000            
            
            L_Naive[i] = int(1000*L_Naive[i])/1000
            Max_mixing_N.append((Bound-L_Naive[i])/3)            
            H_Naive[i] = int(1000*H_Naive[i])/1000  
            
            
    
        Mix_df = {'tau':TAU , 'hi':H_Imbalanced , 'mi':Max_mixing_I,'hg':H_Greedy,'mg':Max_mixing_G,
               'hn':H_Naive,'mn': Max_mixing_N}   
        
        with open(File_name + '/' +'Mix_dict.pkl', 'wb') as f:
            pickle.dump(Mix_df, f)






        df = pd.DataFrame({'Tau':TAU,
            ' Latancy(Im)': L_Imbalanced,
                              'Anonymity(Im)': H_Imbalanced,
            ' Latancy(Gr)': L_Greedy,
                              'Anonymity(Gr)': H_Greedy,
            ' Latancy(N)': L_Naive,
                              'Anonymity(N)': H_Naive                                                           
                              })

        df.to_csv(File_name + '/' +'Analytic_Detailed%d.csv'%((self.Diversify+self.Parameter)*(round(1000*self.mu))), index=False)   


    def Entropy_Latency_Simulation(self,Bound):
        File_name = 'Maximum_Mixing_Delays' +str(round(1000*Bound))
        import numpy as np
        import json
        with open(File_name + '/' +'M_Detailed.json','r') as dicts:
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
        # Open the file to load the dictionary
        with open(File_name + '/' +'Mix_dict.pkl', 'rb') as f:
            Delay_dict = pickle.load(f)        

        for different_tau in range(len(Dictionaries['Itr1'])):
            index = different_tau + 1
###########################################Imbalane approach##############################            
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []  
            self.d1 = Delay_dict['mi'][different_tau]
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
            self.d1 = Delay_dict['mg'][different_tau]
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
            self.d1 = Delay_dict['mn'][different_tau]            
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
        for Tau in np.arange(0,1.01,0.1):            
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
        Delay_dict['HI'] = Entropy_tau_Imbalance
        Delay_dict['HG'] = Entropy_tau_Greedy_Balance        
        Delay_dict['HN'] = Entropy_tau_Naive_Balance     
        with open(File_name + '/' +'Delay_dict.pkl', 'wb') as f:
            pickle.dump(Delay_dict, f)        
        import json
        dics = json.dumps(df)
        with open(File_name + '/' +'M_Detailed_Sim%d.json'%((self.Diversify+self.Parameter)*(self.d1*10000+self.H_N)),'w') as df_sim:
            json.dump(dics,df_sim)        




