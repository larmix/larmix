# -*- coding: utf-8 -*-
"""
Bridge: A bridge attaches two disjoint places, you may wonder how this may come in useful here. Generally, this class provides us with many dictionaries
with which we aim to simulate the mix net. In addition to the mentioned advantages of this class, it includes many helpful analyses for the
analytic approaches.
"""


def Path_Fraction(a,b,c,Dict,W):
    Term = 0
    if len(a) !=0 and len(b) !=0 and len(c) !=0:
        for item1 in a:
            for item2 in b:
                for item3 in c:
                    Term = Term + (1/W)*(Dict['PM%d' %item1][item2-W-1])*(Dict['PM%d' %item2][item3-2*W-1])
    return Term



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


class Bridge(object):
    import math
    import numpy as np

    def __init__(self,Data_set,Clustering,Diversify,Routing,Balancing,N,num_Clusters,
                 Speed_Function,tau,H_parameter,Decimal_A,Algorithm,Layers,mu,strategy,Frc,Goal,Continent,NYM = False,RIPE = True ):
        self.Continent = Continent        
        self.strategy = strategy
        self.NYM = NYM
        self.RIPE = RIPE
        self.G = Goal

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

                    if self.RIPE:
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
    
#############################Preliminary Exps##############################################
####################################Set1###################################################
    def Entropy_Latency_VS_Tau(self,Iterations):
        if self.Diversify ==0:
            File_name = 'Diversification_Basic_EXP' + self.G
        elif self.Diversify ==1:
            File_name = 'Random_Basic_EXP' +  self.G
        elif self.Diversify == 2:
            File_name = 'WC_Basic_EXP' +  self.G
        import os
         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))        
    
            
            
        import numpy as np   
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
            for Tau in np.arange(0,1.01,0.2):
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
        with open(File_name + '/' + 'Dict_save.json','w') as dicts_file:
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
        df = pd.DataFrame({'Tau':TAU,
            ' Latancy(Im)': L_Imbalanced,
                              'Anonymity(Im)': H_Imbalanced,
            ' Latancy(Gr)': L_Greedy,
                              'Anonymity(Gr)': H_Greedy,
            ' Latancy(N)': L_Naive,
                              'Anonymity(N)': H_Naive                              
                              
                              })
        

        df.to_csv(File_name + '/' + 'Analytic.csv', index=False)             

       
###############################################Set2########################################
###########################################Change the number of clusters##################################
    def Entropy_Latency_VS_Clusters(self,Iteration,Tau):
        File_name = 'Num_Clusters'+ str(round(self.mu*1000))+'tau'+str(round(10*Tau))
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))              
        import statistics
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance        
        self.tau = Tau
        self.OneTime = False
        H_Imbalanced = []
        L_Imbalanced = []
        H_Greedy = []
        L_Greedy = []        
        H_Naive = []
        L_Naive = []    
        
        num_Clusters = []        
        corrupted_Mix = {}
        for k in range(1,self.N+1):
            corrupted_Mix['PM%d' %k] = False
        Dictionaries = {}
        INTRY = 1
        NC_ = [2,5,20,40,100,300]
        for I in NC_:
            self.K = I
            H0 = []
            H1 = []
            H2 = []
            L0 = []
            L1 = []
            L2 = []     
            Dicts = {}
            current_dic = 1
            for ITr in range(Iteration):
                Mix_Dict ={}
                Balanced_Greddy_Mixes = {}
                Balanced_Naive_Mixes = {}                

                clusters = Clustering(self.data,self.Clustering,self.K,3,self.Corruption)

                arrangment = Mix_Arrangements(clusters.Mixes,0,clusters.Labels,clusters.Centers,self.Corruption,self.frc,self.OneTime)

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
            
                self.Distance = Distances
                self.Latency = Latency
    
                A,B,C = self.Rout_and_Balance()
            
                self.I_Distributions = A
            
                self.Gready_Balanced_Distributions = B
            
                self.Naive_Balanced_Distributions  =  C
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
                
                H0.append(H[0])
                L0.append(L[0])

                H1.append(H[1])
                L1.append(L[1])

                H2.append(H[2])
                L2.append(L[2])
            Dictionaries['NC%d'%INTRY] = Dicts
            INTRY = INTRY + 1

            H_Imbalanced.append(statistics.median(H0))

            L_Imbalanced.append(statistics.median(L0))           
            
            H_Greedy.append(statistics.median(H1))
            L_Greedy.append(statistics.median(L1))     
    
            H_Naive.append(statistics.median(H2))
            L_Naive.append(statistics.median(L2))   
            
            num_Clusters.append(I) 
            
##########################Save dictionaries for the simulation part######################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'Dict_save_clusters.json','w') as dicts_file:
            json.dump(dicts,dicts_file) 
##################################Save the data we got##################             
        data_clusters_analytic = {'L_Im':L_Imbalanced,'H_Im': H_Imbalanced,'L_Gr':L_Greedy,
                                  'H_Gr':H_Greedy,'L_Na':L_Naive,'H_Na':H_Naive}
        
        data_dic = json.dumps(data_clusters_analytic)
        with open(File_name + '/' + 'save_data_changing_clusters.json','w') as data_clusters:
            json.dump(data_dic,data_clusters)

##############Plot the data##########################################       
            
            
        from Plot import PLOT
        Y = [H_Naive, H_Greedy, H_Imbalanced]
        D = ['Naive Balanced','Greedy Balanced','Low Latency(Imbalance)']
        X_Label = 'Number of Clusters'
        Y_Label = 'Entropy(bit)'
        Name = File_name + '/' + 'cluster_Enteropy.png'
        PLT = PLOT(num_Clusters,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True)
        Y = [L_Naive, L_Greedy, L_Imbalanced]
        Y_Label = 'Latency (sec)'
        Name = File_name + '/' + 'Cluster_Latency.png'
        PLT = PLOT(num_Clusters,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True)
        
##################################Changing the number of N##########################################################        
    def NetworkSize(self,Iteration,tau):
        File_name = 'NetworkSize'+ str(tau)
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))               
        import statistics
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance  
        import numpy as np
        self.tau = tau
        H_Imbalanced = []
        L_Imbalanced = []
        H_Greedy = []
        L_Greedy = []        
        H_Naive = []
        L_Naive = []           
        num_frac = []        
        corrupted_Mix = {}
        for k in range(1,self.N+1):
            corrupted_Mix['PM%d' %k] = False        
        clusters = Clustering(self.data,self.Clustering,self.K,3,self.Corruption)
        CLstMix = []
        Lls = []
        for i in range(self.N):
            CLstMix.append(clusters.Mixes[i,:].tolist())
        for j in range(self.K):
            Lls.append(clusters.Labels[j])
            
        DicLabels = {}
        DicLabels['L'] = Lls
        self.OneTime = False
        Dictionaries = {}
        INTRY = 1
        for I in range(5):        
            self.frac = 0.2+0.2*I 
            H0 = []
            L0 = []
            H1 = []
            L1 = []
            H2 = []
            L2 = []
            Dicts_Frac = {}
            current_dic = 1             
            for K in range(Iteration):
                if not I==0:
                    del arrangment
                    del LatencyDistance
                MC_ =np.zeros((self.N,3))
                for i in range(self.N):
                    MC_[i,:] =CLstMix[i]
                LC_ = []
                for j in range(self.K):
                    LC_.append(DicLabels['L'][j])

                arrangment = Mix_Arrangements(MC_,0,LC_,clusters.Centers,self.Corruption,self.frac,self.OneTime)

                LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)
                if True:
                    if self.RIPE:
                        Distances = LatencyDistance.Topology_to_Distance()
                        Latency = LatencyDistance.RIPE_Latency
            
                self.Distance = Distances
                self.Latency = Latency
    
                A,B,C = self.Rout_and_Balance()
            
                self.I_Distributions = A
            
                self.Gready_Balanced_Distributions = B
            
                self.Naive_Balanced_Distributions  =  C
            
                H,L = self.Make_the_Analysis()

                H0.append(H[0])
                L0.append(L[0])

                H1.append(H[1])
                L1.append(L[1])

                H2.append(H[2])
                L2.append(L[2])

                Mix_Dict ={}
                Balanced_Greddy_Mixes = {}
                Balanced_Naive_Mixes = {}
                
                counter = 0
                for jj in range( self.L-1):
                    for ii in range(round(self.W*self.frac)):
                        counter = counter + 1
                        Mix_Dict['PM%d' %counter] = self.I_Distributions[ii,:,jj].tolist()
                        Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[ii,:,jj].tolist()
                        Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[ii,:,jj].tolist()
       
                Dict={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                      'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': corrupted_Mix,'Latency': self.Latency}
                Dicts_Frac['Dic%d' %current_dic] = Dict
                current_dic = current_dic + 1
            H_Imbalanced.append(statistics.median(H0))                
            L_Imbalanced.append(statistics.median(L0))           
            
            H_Greedy.append(statistics.median(H1))
            L_Greedy.append(statistics.median(L1))     
    
            H_Naive.append(statistics.median(H2))
            L_Naive.append(statistics.median(L2))             
            num_frac.append(self.frac) 
            Dictionaries['NC%d'%INTRY] = Dicts_Frac
            INTRY = INTRY + 1
##########################Save dictionaries for the simulation part###########################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'Dict_save_nodes.json','w') as dicts_file:
            json.dump(dicts,dicts_file) 
##################################Save the data we got########################################             
        data_nodes_analytic = {'L_Im':L_Imbalanced,'H_Im': H_Imbalanced,'L_Gr':L_Greedy,
                                  'H_Gr':H_Greedy,'L_Na':L_Naive,'H_Na':H_Naive}
        
        data_dic = json.dumps(data_nodes_analytic)
        with open(File_name + '/' + 'save_data_changing_nodes.json','w') as data_nodes:
            json.dump(data_dic,data_nodes)



        
###########################################Set3##################################################
##################################Changing the number of N##########################################################        
    def Entropy_Latency_VS_Fravtion_Of_N(self,Iteration,tau):
        File_name = 'Num_mix' + str(round(self.mu*1000)) + 'Diversify' + str(round(self.Diversify))
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))               
        import statistics
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance  
        import numpy as np
        self.tau = tau
        H_Imbalanced = []
        L_Imbalanced = []
        H_Greedy = []
        L_Greedy = []        
        H_Naive = []
        L_Naive = []           
        num_frac = []        
        corrupted_Mix = {}
        for k in range(1,self.N+1):
            corrupted_Mix['PM%d' %k] = False        
        clusters = Clustering(self.data,self.Clustering,self.K,3,self.Corruption)
        CLstMix = []
        Lls = []
        for i in range(self.N):
            CLstMix.append(clusters.Mixes[i,:].tolist())
        for j in range(self.K):
            Lls.append(clusters.Labels[j])
            
        DicLabels = {}
        DicLabels['L'] = Lls
        self.OneTime = False
        Dictionaries = {}
        INTRY = 1
        for I in range(5):        
            self.frac = 0.2+0.2*I 
            H0 = []
            L0 = []
            H1 = []
            L1 = []
            H2 = []
            L2 = []
            Dicts_Frac = {}
            current_dic = 1             
            for K in range(Iteration):
                if not I==0:
                    del arrangment
                    del LatencyDistance
                MC_ =np.zeros((self.N,3))
                for i in range(self.N):
                    MC_[i,:] =CLstMix[i]
                LC_ = []
                for j in range(self.K):
                    LC_.append(DicLabels['L'][j])

                arrangment = Mix_Arrangements(MC_,0,LC_,clusters.Centers,self.Corruption,self.frac,self.OneTime)

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
            
                self.Distance = Distances
                self.Latency = Latency
    
                A,B,C = self.Rout_and_Balance()
            
                self.I_Distributions = A
            
                self.Gready_Balanced_Distributions = B
            
                self.Naive_Balanced_Distributions  =  C
            
                H,L = self.Make_the_Analysis()

                H0.append(H[0])
                L0.append(L[0])

                H1.append(H[1])
                L1.append(L[1])

                H2.append(H[2])
                L2.append(L[2])

                Mix_Dict ={}
                Balanced_Greddy_Mixes = {}
                Balanced_Naive_Mixes = {}
                
                counter = 0
                for jj in range( self.L-1):
                    for ii in range(round(self.W*self.frac)):
                        counter = counter + 1
                        Mix_Dict['PM%d' %counter] = self.I_Distributions[ii,:,jj].tolist()
                        Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[ii,:,jj].tolist()
                        Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[ii,:,jj].tolist()
       
                Dict={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                      'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': corrupted_Mix,'Latency': self.Latency}
                Dicts_Frac['Dic%d' %current_dic] = Dict
                current_dic = current_dic + 1
            H_Imbalanced.append(statistics.median(H0))                
            L_Imbalanced.append(statistics.median(L0))           
            
            H_Greedy.append(statistics.median(H1))
            L_Greedy.append(statistics.median(L1))     
    
            H_Naive.append(statistics.median(H2))
            L_Naive.append(statistics.median(L2))             
            num_frac.append(self.frac) 
            Dictionaries['NC%d'%INTRY] = Dicts_Frac
            INTRY = INTRY + 1
##########################Save dictionaries for the simulation part###########################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'Dict_save_nodes.json','w') as dicts_file:
            json.dump(dicts,dicts_file) 
##################################Save the data we got########################################             
        data_nodes_analytic = {'L_Im':L_Imbalanced,'H_Im': H_Imbalanced,'L_Gr':L_Greedy,
                                  'H_Gr':H_Greedy,'L_Na':L_Naive,'H_Na':H_Naive}
        
        data_dic = json.dumps(data_nodes_analytic)
        with open(File_name + '/' + 'save_data_changing_nodes.json','w') as data_nodes:
            json.dump(data_dic,data_nodes)


##############################################################################################
##############################################################################################        
###############################################Set4########################################
###########################################Clustering matters##################################
    def Entropy_Latency_VS_clustering_methodes(self,Iteration,Tau,Limitation):
        File_name = 'Clustering_Methods'
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))              
        if Limitation:
            Methods = ['kmeans','kmedoids']
        else: 
            Methods = ['kmeans','kmedoids','FCM']
        import statistics
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance        
        self.tau = Tau
        self.OneTime = False
        H_Imbalanced = []
        L_Imbalanced = []
        H_Greedy = []
        L_Greedy = []        
        H_Naive = []
        L_Naive = []    
        
        num_Clusters = []        
        corrupted_Mix = {}
        for k in range(1,self.N+1):
            corrupted_Mix['PM%d' %k] = False
        Dictionaries = {}
        INTRY = 1
        for approach in Methods:
            self.Clustering = approach
            H0 = []
            H1 = []
            H2 = []
            L0 = []
            L1 = []
            L2 = []     
            Dicts = {}
            current_dic = 1
            for ITr in range(Iteration):
                Mix_Dict ={}
                Balanced_Greddy_Mixes = {}
                Balanced_Naive_Mixes = {}                

                clusters = Clustering(self.data,self.Clustering,self.K,3,self.Corruption)

                arrangment = Mix_Arrangements(clusters.Mixes,0,clusters.Labels,clusters.Centers,self.Corruption,self.frc,self.OneTime)

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
            
                self.Distance = Distances
                self.Latency = Latency
    
                A,B,C = self.Rout_and_Balance()
            
                self.I_Distributions = A
            
                self.Gready_Balanced_Distributions = B
            
                self.Naive_Balanced_Distributions  =  C
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
                
                H0.append(H[0])
                L0.append(L[0])

                H1.append(H[1])
                L1.append(L[1])

                H2.append(H[2])
                L2.append(L[2])
            Dictionaries['NC%d'%INTRY] = Dicts
            INTRY = INTRY + 1

            H_Imbalanced.append(statistics.median(H0))

            L_Imbalanced.append(statistics.median(L0))           
            
            H_Greedy.append(statistics.median(H1))
            L_Greedy.append(statistics.median(L1))     
    
            H_Naive.append(statistics.median(H2))
            L_Naive.append(statistics.median(L2))   
            

            
##########################Save dictionaries for the simulation part######################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'Dict_save_Methods.json','w') as dicts_file:
            json.dump(dicts,dicts_file) 
##################################Save the data we got##################             
        data_clusters_analytic = {'L_Im':L_Imbalanced,'H_Im': H_Imbalanced,'L_Gr':L_Greedy,
                                  'H_Gr':H_Greedy,'L_Na':L_Naive,'H_Na':H_Naive}
        
        data_dic = json.dumps(data_clusters_analytic)
        with open(File_name + '/' + 'save_data_changing_Methods.json','w') as data_clusters:
            json.dump(data_dic,data_clusters)

##############Plot the data##########################################       
            
            
        from Plot import PLOT
        Y = [H_Naive, H_Greedy, H_Imbalanced]
        D = ['Naive Balanced','Greedy Balanced','Low Latency(Imbalance)']
        X_Label = 'Methods'
        Y_Label = 'Entropy(bit)'
        Name = File_name + '/' + 'Methods_Enteropy.png'
        PLT = PLOT(Methods,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True)
        Y = [L_Naive, L_Greedy, L_Imbalanced]
        Y_Label = 'Latency (sec)'
        Name = File_name + '/' + 'Methods_Latency.png'
        PLT = PLOT(Methods,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True)
         

