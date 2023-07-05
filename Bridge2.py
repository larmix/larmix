# -*- coding: utf-8 -*-
"""
Adversarial Bridge 
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


class Adversarial_Bridge(object):
    import math
    import numpy as np

    def __init__(self,Data_set,Clustering,Diversify,Routing,Balancing,N,num_Clusters,
                 Speed_Function,tau,H_parameter,Decimal_A,Algorithm,Layers,mu,strategy,Frc,Goal,NYM = False,RIPE=False, EXC = False):
        self.Exc = EXC
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
        self.NYM = NYM
        
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
    
    def Greedy_For_Fairness(self,Dict,C):
        import numpy as np
        C_i = int(C/3) #Minimun mix nodes an adversary can have while they equally distribute them among the layers is derived in this way.
        if nCr(self.W,C_i) < 60*self.N+1:
            import timeit        
            Num_mix_nodes = []
            for i in range(1,self.W+1):
                Num_mix_nodes.append(i)                
            WL = findsubsets(set(Num_mix_nodes), C_i)            
        else:
            # There is no need to check all possible cases as it's gonna be taking ages we just check a limmited sets
            #Just when N is large enough            
            import random
            WL = []
            import random
            LIs = []
            for j in range(self.W):
                LIs.append( j+1)            
            while len(WL)<60*self.N+1:

                RNDM = tuple(random.sample(LIs, C_i))
                if not (RNDM in WL):
                    WL.append(RNDM)
            WL = set(WL)               
#When the intial set has been fixed, it times for saking the best case 
        Max = 0
        for itemm in WL:
            c = 3*C_i           
            item = list(itemm)
            X = np.zeros((C_i,self.W))
            j = 0
            for terms in item:
                X[j,:] = Dict['PM%d' %terms]
                j = j+1
            X_SUM = np.sum(X , axis = 0)
            x_sum = X_SUM.tolist()

            Index_x = sort_index(x_sum , C_i)

            Y = np.zeros((C_i,self.W))
            j = 0
            for terms in Index_x:
                Y[j,:] = Dict['PM%d' %(terms+1+self.W)]
                j = j +1

            Y_SUM = np.sum(Y , axis = 0)
            y_sum = Y_SUM.tolist()

            Index_y = sort_index(y_sum,C_i)
#We are about to complete the most effective choice of corrupted mix nodes
#Just note sometimes the budget is not a factor of 3 so we may try 1 or 2 times more to
#distinguish remained corrupted mix nodes        
            while(c < C):
                Par = 0
                for m in range(1,self.W+1):
                    if not (m in item ):
                        parameter = 0
                        for TRm in Index_x:
                            parameter = parameter + Dict['PM%d' %(m)][int(TRm)]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
                                            
                for m in range(self.W+1,2*self.W+1):
                    if not ((m-1-self.W) in Index_x ):
                        parameter = 0
                        for TRm1 in Index_y:
                            parameter = parameter + 0.5*Dict['PM%d' %(m)][int(TRm1)]
                        for TRm2 in item:
                            parameter = parameter + 0.5*Dict['PM%d' %(int(TRm2))][m-1-self.W]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
            
                
                for m in range(2*self.W+1,3*self.W+1):
                    if not ((m-1-2*self.W) in Index_y ):
                        parameter = 0
                        for TRm in Index_x:
                            parameter = parameter + Dict['PM%d' %(int(TRm)+self.W+1)][m-2*self.W-1]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
                if Inx < self.W+1:
                    item.append(Inx)
                elif self.W< Inx <2*self.W+1:
                    Index_x.append(Inx -1 -self.W)
                elif 2*self.W < Inx:
                    Index_y.append(Inx-2*self.W-1)
                c = c+1
            Term1 = 0
            for item1 in item:
                for item2 in Index_x:
                    for item3 in Index_y:
                        Term1 = Term1 + (1/self.W)*(Dict['PM%d' %item1][item2])*(Dict['PM%d' %(self.W+1+item2)][item3])
            if Max<Term1:
                Max = Term1

        return Max    
    
    def Greedy_For_Mix_nodes(self,Dict,C,C_L):
        a_dis = [0]
        b_dis = [0]
        c_dis = [0]
        Max = 0


        for item in(C_L): #C_L is a list including all sets with size of three chosen from the mix nodes set
            a = []
            b = []
            c = []
            I = list(item)
            for term in I:
                if term < self.W+1:
                    a.append(term)
                elif self.W < term <2*self.W+1:
                    b.append(term)
                elif term>2*self.W:
                    c.append(term)
            Term1 = 0

            if len(a) !=0 and len(b) !=0 and len(c) !=0:
                for item1 in a:
                    for item2 in b:
                        for item3 in c:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %item1][item2-self.W-1])*(Dict['PM%d' %item2][item3-2*self.W-1])

            if Term1 > Max:
                a_dis[0] = item1
                b_dis[0] = item2
                c_dis[0] = item3
                Max = Term1

        for i in range(4,C+1):

            Par1 = 0
            for m in range(1,self.W+1):        
                if not (m in a_dis):
                    Parameter1 = Path_Fraction(a_dis+[m],b_dis,c_dis,Dict,self.W)

                    if Parameter1>Par1:
                        Ind1 = m
                        Par1 = Parameter1


            for m in range(self.W+1,2*self.W+1):       
                if not (m in b_dis):
                    Parameter1 = Path_Fraction(a_dis,b_dis + [m],c_dis,Dict,self.W)
                            
                    if Parameter1>Par1:
                        Ind1 = m
                        Par1 = Parameter1

            for m in range(2*self.W+1,3*self.W+1):
                if not (m in c_dis):
                    Parameter1 = Path_Fraction(a_dis,b_dis,c_dis+[m],Dict,self.W)
                    
                    if Parameter1>Par1:
                        Ind1 = m
                        Par1 = Parameter1


                               
            if Ind1 < self.W+1:
                a_dis.append(Ind1)
            elif self.W< Ind1< 2*self.W+1:
                b_dis.append(Ind1)
            elif Ind1>2*self.W:
                c_dis.append(Ind1)
                                
        TRM1 = Path_Fraction(a_dis,b_dis,c_dis,Dict,self.W)
        return TRM1



    def Best_case_chosen_of_corrupted_paths(self,Dict,C,C_L):
        FCP1 = self.Greedy_For_Fairness(Dict, C)
        FCP2 = self.Greedy_For_Mix_nodes(Dict, C, C_L)
        if FCP1>FCP2:
            return FCP1
        else:
            return FCP2
    
    
    def Greedy_For_Paths(self,Dict,C):
        import numpy as np
        Lists  = []
        Relays = []
        for i in range(self.W):
            for j in range(self.W):
                for k in range(self.W):
                    P = (1/self.W)*(Dict['PM%d' %(i+1)][j])*(Dict['PM%d' %(self.W+j+1)][k])
                    Lists.append( [i+1,self.W+j+1,2*self.W+k+1,P ])
        array = np.array(Lists)
        Sort_array = array[np.argsort(array[:, 3])]
        (a,b) = np.shape(Sort_array)
        c = 0
        for row in  range(a):
            if c < C:
                for colomn in range(b-1):
                    if not (Sort_array[row,colomn] in Relays):
                        Relays.append(Sort_array[row,colomn])
                        c = c+1
                        if not c<C:
                            break
        a = []
        b = []
        c = []        
        for term in(Relays):
            if term < self.W+1:
                a.append(term)
            elif self.W < term <2*self.W+1:
                b.append(term)
            elif term>2*self.W:
                c.append(term)
        Term1 = 0
        if len(a) !=0 and len(b) !=0 and len(c) !=0:
            for item1 in a:
                for item2 in b:
                    for item3 in c:
                        Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                        
        return Term1    
    
    
    def exhaustive_search(self,Dict,C_LIST):
        
        Max = 0
        for item in(C_LIST):
            a = []
            b = []
            c = []
            I = list(item)
            for term in I:
                if term < self.W+1:
                    a.append(term)
                elif self.W < term <2*self.W+1:
                    b.append(term)
                elif term>2*self.W:
                    c.append(term)
            Term1 = 0

            if len(a) !=0 and len(b) !=0 and len(c) !=0:
                for item1 in a:
                    for item2 in b:
                        for item3 in c:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %item1][item2-self.W-1])*(Dict['PM%d' %item2][item3-2*self.W-1])

            if Term1 > Max:
                Max = Term1
        return Max
            
    
    def test(self,cc,TAU,TEST): 
                          
        self.tau = TAU

  
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
                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j]
                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j]
                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j]
       
        if TEST == 'EXH':
            
            import timeit


        
            Num_mix_nodes = []

            for i in range(1,self.N+1):
                Num_mix_nodes.append(i)
                
            C_LIST = findsubsets(set(Num_mix_nodes), cc)
            
            itemA = self.exhaustive_search(Mix_Dict, C_LIST)
            itemB = self.exhaustive_search(Balanced_Greddy_Mixes, C_LIST)            
            itemC = self.exhaustive_search(Balanced_Naive_Mixes, C_LIST)
        elif TEST == 'Gr_Pa':
            
            
            itemA = self.Greedy_For_Paths(Mix_Dict, cc)
            itemB = self.Greedy_For_Paths(Balanced_Greddy_Mixes, cc)            
            itemC = self.Greedy_For_Paths(Balanced_Naive_Mixes, cc)


        elif TEST == 'Gr_F':
            
            import timeit


        

            
            itemA = self.Greedy_For_Numbers(Mix_Dict,cc)
            itemB = self.Greedy_For_Numbers(Balanced_Greddy_Mixes,cc)            
            itemC = self.Greedy_For_Numbers(Balanced_Naive_Mixes,cc)
        elif TEST == 'Gr_Pr':
            
            import timeit


        
            Num_mix_nodes = []

            for i in range(1,self.N+1):
                Num_mix_nodes.append(i)
                
            C_L = findsubsets(set(Num_mix_nodes), 3)
            itemA = self.Greedy_For_Probabilities(Mix_Dict,cc,C_L)
            itemB = self.Greedy_For_Probabilities(Balanced_Greddy_Mixes,cc, C_L)            
            itemC = self.Greedy_For_Probabilities(Balanced_Naive_Mixes,cc, C_L)
        return itemA,itemB,itemC

    
    
    def FCP(self,C,Dict):
        a = []
        b = []
        c = []
        for i in range(1,self.W+1):
            if C['PM%d' %i] == True:
                a.append(i)                
        for i in range(self.W+1,2*self.W+1):
            if C['PM%d' %i] == True:
                b.append(i)            
        
        for i in range(2*self.W+1,3*self.W+1):
            
            if C['PM%d' %i] == True:
                c.append(i)
        Term1 = 0
        if len(a) !=0 and len(b) !=0 and len(c) !=0:
            for item1 in a:
                for item2 in b:
                    for item3 in c:
                        Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1])                         
        return Term1
   



    def Corruptions_scenarios(self,Strategy,c_m):                      
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance        
        from Corruption import corruptedMix
        if Strategy == 2:
            import numpy as np
            MM = np.zeros((self.N,3))
            LM = 0            
            Class_CNs = corruptedMix(self.data,c_m,MM,LM,Strategy)
            Class_CNs.corrupted_mix_nodes()
            C10 = Class_CNs.CNs
            clusters = Clustering(self.data,self.Clustering,self.K,3,C10)
            C_C = clusters.mapping()
            clusters.Data_plt()       
            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C_C,self.frc,True)
            
            Corruptted = arrangment.mapping()
        
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
            
            if Strategy == 6:
                clusters = Clustering(self.data,self.Clustering,self.K,3,0)
                arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,0,self.frc,False)
     
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
                    Corruptted = 0
                
                
            else:
                import numpy as np


                clusters = Clustering(self.data,self.Clustering,self.K,3,0)

                clusters.Data_plt()
                Class_CNs = corruptedMix(self.data,c_m,clusters.Mixes,clusters.Labels,Strategy)
                Class_CNs.corrupted_mix_nodes()
                C11 = Class_CNs.CNs        
                arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
        

                Corruptted = arrangment.mapping()
        
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
        return Distances, Latency, Corruptted

     
                       
    def Fraction_of_corrupted_Path(self,Itr):#We tend to draw fraction of corrupted path V.S. different value of \tau
        if self.Diversify ==0:
            File_name = 'Diversification_FCP'
        elif self.Diversify ==1:
            File_name = 'Random_FCP'
        elif self.Diversify == 2:
            File_name = 'WC_FCP'
        import os
         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name)) 
        TAU = [0,0.2,0.4,0.6,0.8,1] #put your desired value of \tau in this list, note as we are limmitted in showing the
        #Box plots 7 value of \tau is the maximum allowed
        
        if self.Diversify ==0:
            if self.Exc:
                Strategy = [2,4,1,5]
            else:
                Strategy = [6,2,4,1,5]#We have diffrent strategy regarding the relay adversaries:
        else:
            if self.Exc:
                Strategy = [2,1,5]
            else:
                Strategy = [6,2,1,5]
        
        #1:Randomly picked out mix nodes 2:The adversaries owning mix nodes in adistinct location
        #3:-- 4:The adversary farly choses mix nodes from different clusters
        #5:The worst case scenario 
        #6:The best case scenario
        c = 0.2#Fraction of corrupted mix nodes varies from 0 to 30 percent
        FCPIM = []#FCP stabds for Fraction of Corrupted Path and Im represents Imbalance approach(Low Latency)
        #FCPIM includes FCP of diffrent strtegies whilke we considered Low Latency routing,FCPGR and FCPNA do the same 
        #when we agree to call 'Greedy' as GR and 'Naive' as NA.
        FCPGR = []
        FCPNA = []
        Dictionaries = {} #We save all derived PDFs in this dictioary which comes in useful for the simulation part.
        INDEX = 1 #The dictionary needs index starting from 1.
        for T in TAU:#We have a loop to exmine changing \tau values
            if T ==0:
                T == 0.01
            #The three following lists are as same as above lists except for fixing the value of tau for them.
            F_C_P_Im = []
            F_C_P_Gr = []
            F_C_P_Na = [] 
            Dicts = {} #Regarding a fixed \tau we'll have a dictiionary saving the PDFs, IT will add to the Dictionaries
            index_dic = 1#The mentioned dictionary has an index beggining from 1
            for I in range(Itr):#After fixing the value of tau we have to allow for some iterations to make sure we avoide 
                #sampling errors
                F_C_P_Im0 = []#These three lists have simmilar difenitions while we fixed one iteration
                F_C_P_Gr0 = []
                F_C_P_Na0 = []
                C_Nodes   = {}
                Dict_s    = {}#In this case we save PDFs of all strategies in this dictionary
                
                for s in Strategy:
                    if not s==6 and not s==5: #For all strategies except for 5 and 6:
                        D,L,C = self.Corruptions_scenarios(s,c)#We derive D:Distances,L:Latencies and C:Corrupted mix nodes from the implied function
                        self.tau      = T
                        self.Distance = D
                        self.Latency  = L
                        #Now is the time for derin=ving the routing aproaches
                        AA,BB,CC= self.Rout_and_Balance()  #Use this function to fulfill our aims          
                        self.I_Distributions = AA     #Low Latency routing       
                        self.Gready_Balanced_Distributions = BB  #Balanced(Greedy)          
                        self.Naive_Balanced_Distributions = CC   #Balanced(Naive) 
                        #Above distributions are gonna be stored respectively in following dictionaries
                        Mix_Dict ={}
                        Balanced_Greddy_Mixes = {}
                        Balanced_Naive_Mixes = {}
                        counter = 0
                        for j in range( self.L-1):#That's how we save them
                            for i in range(self.W):
                                counter = counter + 1
                                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
                        #Now all the info about the chosen strategy will save in following dictionary                              
                        Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}
                        #As you may remember we have lists for saving the FCP, we may do this regarding FCP function
                        F_C_P_Im0.append(self.FCP(C,Mix_Dict))
                        F_C_P_Gr0.append(self.FCP(C,Balanced_Greddy_Mixes))                   
                        F_C_P_Na0.append(self.FCP(C,Balanced_Naive_Mixes))                       
                    else:
                        if s ==6: #if strategy is s = 6                        
                            D,L,C1 = self.Corruptions_scenarios(s,c) #Here we achieve D:Distances and L:Latencies
                            #From The corruption_Scenarios function and c1 is just a mock value
                            from Corruption import corruptedMix 
                            import numpy as np
                            MM = np.zeros((self.N,3))# This is just a function to intilaize following class,we don't use it later on
                            LM = 0    #Mock parameter        
                            Class_CNs = corruptedMix(MM,c,MM,LM,6) #wE MAINLY USE THIS TO take advantage of fits function for deriving the Cnodes         
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                    #The above codes are loking like whay we explained before just the following are added to attain the
                                    #corrupted mix nodes
                            C_GFP_Im,add1 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Mix_Dict)
                            C_GFP_Gr,add2 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Greddy_Mixes)
                            C_GFP_Na,add3 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Naive_Mixes)

                            F_C_P_Im0.append( self.FCP(C_GFP_Im,Mix_Dict))
                            F_C_P_Gr0.append(self.FCP(C_GFP_Gr,Balanced_Greddy_Mixes))                   
                            F_C_P_Na0.append(self.FCP(C_GFP_Na,Balanced_Naive_Mixes))                    
    
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'corrupted_Mix': C_GFP_Im,'Latency': self.Latency}   
                            Dict_s['Strategy%d'%(s+1)]={'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'corrupted_Mix': C_GFP_Gr,'Latency': self.Latency}                             
                            Dict_s['Strategy%d'%(s+2)]={
                            'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C_GFP_Na,'Latency': self.Latency}                             
                        elif s==5:#For sake of the worst case scenario we're gonna use the following codes which are more or less
                            #equal to what we had already,though we may use different function to achieve CNodes
                            from Clustering import Clustering
                            from MixNetArrangment import Mix_Arrangements
                            from Latency import Latency_and_Distance        
                            from Corruption import corruptedMix                            
                            clusters = Clustering(self.data,self.Clustering,self.K,3,0) 
                            C11 = {}
                            for i in range(self.N):
                                j = i +1
                                C11['PM%d' %j] = False
                            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
                            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)
                             
                            if (not self.NYM) and  (not self.RIPE):
                                Distances_ = LatencyDistance.Topology_to_Distance()
                                D = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_)
                                L = LatencyDistance.Distance_to_Latency()
                            else:
                                if self.NYM:
                                    D = LatencyDistance.Topology_to_Distance()
                                    L = LatencyDistance.NYM_Latency
                                elif self.RIPE:
                                    D = LatencyDistance.Topology_to_Distance()
                                    L = LatencyDistance.RIPE_Latency  
                            from Corruption import corruptedMix
                            import numpy as np

                            MM = np.zeros((self.N,3))
                            LM = 0
            
                            Class_CNs = corruptedMix(MM,c,MM,LM,5)
          
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                            CN_ = Class_CNs.Worst_Case(Mix_Dict,False)                            
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': CN_,'Latency': self.Latency}  
                            F_C_P_Im0.append(0)
                            F_C_P_Gr0.append(0)                   
                            F_C_P_Na0.append(0)                            
                            
                            
                Dicts['Dic%d' %index_dic] = Dict_s
                index_dic = index_dic + 1       
                F_C_P_Im.append(F_C_P_Im0)
                F_C_P_Gr.append(F_C_P_Gr0)
                F_C_P_Na.append(F_C_P_Na0)
            Dictionaries['DicT%d' %INDEX] = Dicts
            INDEX = INDEX+1
            M1 = np.matrix(F_C_P_Im) 
            Im = Med((np.transpose(M1)).tolist())
            M2 = np.matrix(F_C_P_Gr)
            Gr = Med((np.transpose(M2)).tolist())            
            M3 = np.matrix(F_C_P_Na)
            Na = Med((np.transpose(M3)).tolist())                
            FCPIM.append(Im)
            FCPGR.append(Gr)
            FCPNA.append(Na)
        Matrix_Im = np.transpose(np.matrix(FCPIM))
        Matrix_Gr = np.transpose(np.matrix(FCPGR))
        Matrix_Na = np.transpose(np.matrix(FCPNA)) 
##################################Save the data for simulations##############################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' + 'Dict_save_corrupted_tau.json','w') as dicts_c_t:
            json.dump(dicts,dicts_c_t)
##################################Save the data we got##################             
        data_corrupted_analytic_change_tau = {'tau':TAU,'Im': Matrix_Im,'Gr':Matrix_Gr,
                                  'Na':Matrix_Na}
############################Save the data we got in this part########################  
        import pickle
        M_Corruption = open(File_name + '/' + 'save_data_changing_corrupted_tau.pkl','wb')
        pickle.dump(data_corrupted_analytic_change_tau, M_Corruption)
        M_Corruption.close()                
##################################################################################################        

        
#################################################################################################
#################################################################################################        
##################################################################################################


                      
    def Fraction_of_corrupted_Path_number_of_Cnodes(self,Itr):#We tend to draw fraction of corrupted path V.S. different value of \tau
        C_NODES = [0.1,0.2,0.3] #put your desired value of \tau in this list, note as we are limmitted in showing the
        #Box plots 7 value of \tau is the maximum allowed
        if self.Diversify ==0:
            File_name = 'Diversification_FCP_CNodes'
        elif self.Diversify ==1:
            File_name = 'Random_FCP_CNodes'
        elif self.Diversify == 2:
            File_name = 'WC_FCP_CNodes'     
        import os
         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))         
        if self.Diversify ==0:
            if self.Exc:
                Strategy = [2,4,1,5]
            else:
                Strategy = [6,2,4,1,5]#We have diffrent strategy regarding the relay adversaries:
        
            
        else:
            
            if self.Exc:
                Strategy = [2,1,5]
            else:
                Strategy = [2,6,1,5]
        T  = 0.56
        
        #1:Randomly picked out mix nodes 2:The adversaries owning mix nodes in adistinct location
        #3:-- 4:The adversary farly choses mix nodes from different clusters
        #5:The worst case scenario 
        #6:The best case scenario

        FCPIM = []#FCP stabds for Fraction of Corrupted Path and Im represents Imbalance approach(Low Latency)
        #FCPIM includes FCP of diffrent strtegies whilke we considered Low Latency routing,FCPGR and FCPNA do the same 
        #when we agree to call 'Greedy' as GR and 'Naive' as NA.
        FCPGR = []
        FCPNA = []
        Dictionaries = {} #We save all derived PDFs in this dictioary which comes in useful for the simulation part.
        INDEX = 1 #The dictionary needs index starting from 1.
        for c in C_NODES:#We have a loop to exmine changing \tau values
            #The three following lists are as same as above lists except for fixing the value of tau for them.
            F_C_P_Im = []
            F_C_P_Gr = []
            F_C_P_Na = [] 
            Dicts = {} #Regarding a fixed \tau we'll have a dictiionary saving the PDFs, IT will add to the Dictionaries
            index_dic = 1#The mentioned dictionary has an index beggining from 1
            for I in range(Itr):#After fixing the value of tau we have to allow for some iterations to make sure we avoide 
                #sampling errors
                F_C_P_Im0 = []#These three lists have simmilar difenitions while we fixed one iteration
                F_C_P_Gr0 = []
                F_C_P_Na0 = []
                C_Nodes   = {}
                Dict_s    = {}#In this case we save PDFs of all strategies in this dictionary
                
                for s in Strategy:
                    if not s==6 and not s==5: #For all strategies except for 5 and 6:
                        D,L,C = self.Corruptions_scenarios(s,c)#We derive D:Distances,L:Latencies and C:Corrupted mix nodes from the implied function
                        self.tau      = T
                        self.Distance = D
                        self.Latency  = L
                        #Now is the time for derin=ving the routing aproaches
                        AA,BB,CC= self.Rout_and_Balance()  #Use this function to fulfill our aims          
                        self.I_Distributions = AA     #Low Latency routing       
                        self.Gready_Balanced_Distributions = BB  #Balanced(Greedy)          
                        self.Naive_Balanced_Distributions = CC   #Balanced(Naive) 
                        #Above distributions are gonna be stored respectively in following dictionaries
                        Mix_Dict ={}
                        Balanced_Greddy_Mixes = {}
                        Balanced_Naive_Mixes = {}
                        counter = 0
                        for j in range( self.L-1):#That's how we save them
                            for i in range(self.W):
                                counter = counter + 1
                                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
                        #Now all the info about the chosen strategy will save in following dictionary                              
                        Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}
                        #As you may remember we have lists for saving the FCP, we may do this regarding FCP function
                        F_C_P_Im0.append(self.FCP(C,Mix_Dict))
                        F_C_P_Gr0.append(self.FCP(C,Balanced_Greddy_Mixes))                   
                        F_C_P_Na0.append(self.FCP(C,Balanced_Naive_Mixes))                       
                    else:
                        if s ==6: #if strategy is s = 6                        
                            D,L,C1 = self.Corruptions_scenarios(s,c) #Here we achieve D:Distances and L:Latencies
                            #From The corruption_Scenarios function and c1 is just a mock value
                            from Corruption import corruptedMix 
                            import numpy as np
                            MM = np.zeros((self.N,3))# This is just a function to intilaize following class,we don't use it later on
                            LM = 0    #Mock parameter        
                            Class_CNs = corruptedMix(MM,c,MM,LM,6) #wE MAINLY USE THIS TO take advantage of fits function for deriving the Cnodes         
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                    #The above codes are loking like whay we explained before just the following are added to attain the
                                    #corrupted mix nodes
                            C_GFP_Im,add1 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Mix_Dict)
                            C_GFP_Gr,add2 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Greddy_Mixes)
                            C_GFP_Na,add3 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Naive_Mixes)

                            F_C_P_Im0.append( self.FCP(C_GFP_Im,Mix_Dict))
                            F_C_P_Gr0.append(self.FCP(C_GFP_Gr,Balanced_Greddy_Mixes))                   
                            F_C_P_Na0.append(self.FCP(C_GFP_Na,Balanced_Naive_Mixes))                    
    
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'corrupted_Mix': C_GFP_Im,'Latency': self.Latency}   
                            Dict_s['Strategy%d'%(s+1)]={'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'corrupted_Mix': C_GFP_Gr,'Latency': self.Latency}                             
                            Dict_s['Strategy%d'%(s+2)]={
                            'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C_GFP_Na,'Latency': self.Latency}                             
                        elif s==5:#For sake of the worst case scenario we're gonna use the following codes which are more or less
                            #equal to what we had already,though we may use different function to achieve CNodes
                            from Clustering import Clustering
                            from MixNetArrangment import Mix_Arrangements
                            from Latency import Latency_and_Distance        
                            from Corruption import corruptedMix                            
                            clusters = Clustering(self.data,self.Clustering,self.K,3,0) 
                            C11 = {}
                            for i in range(self.N):
                                j = i +1
                                C11['PM%d' %j] = False
                            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
                            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function,self.G,self.NYM,self.RIPE)
                             
                            if (not self.NYM) and  (not self.RIPE):
                                Distances_ = LatencyDistance.Topology_to_Distance()
                                D = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_)
                                L = LatencyDistance.Distance_to_Latency()
                            else:
                                if self.NYM:
                                    D = LatencyDistance.Topology_to_Distance()
                                    L = LatencyDistance.NYM_Latency
                                elif self.RIPE:
                                    D = LatencyDistance.Topology_to_Distance()
                                    L = LatencyDistance.RIPE_Latency                             
 
                            from Corruption import corruptedMix
                            import numpy as np

                            MM = np.zeros((self.N,3))
                            LM = 0
            
                            Class_CNs = corruptedMix(MM,c,MM,LM,5)
          
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                            C = Class_CNs.Worst_Case(Mix_Dict,False)                            
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}  
                            F_C_P_Im0.append(0)
                            F_C_P_Gr0.append(0)                   
                            F_C_P_Na0.append(0)                            
                            
                            
                Dicts['Dic%d' %index_dic] = Dict_s
                index_dic = index_dic + 1       
                F_C_P_Im.append(F_C_P_Im0)
                F_C_P_Gr.append(F_C_P_Gr0)
                F_C_P_Na.append(F_C_P_Na0)
            Dictionaries['DicT%d' %INDEX] = Dicts
            INDEX = INDEX+1
            M1 = np.matrix(F_C_P_Im) 
            Im = Med((np.transpose(M1)).tolist())
            M2 = np.matrix(F_C_P_Gr)
            Gr = Med((np.transpose(M2)).tolist())            
            M3 = np.matrix(F_C_P_Na)
            Na = Med((np.transpose(M3)).tolist())                
            FCPIM.append(Im)
            FCPGR.append(Gr)
            FCPNA.append(Na)
        Matrix_Im = np.transpose(np.matrix(FCPIM))
        Matrix_Gr = np.transpose(np.matrix(FCPGR))
        Matrix_Na = np.transpose(np.matrix(FCPNA)) 
##################################Save the data for simulations##############################
        import json
        dicts = json.dumps(Dictionaries)
        with open(File_name + '/' +'Dict_save_corrupted_Cnodes.json','w') as dicts_c_t:
            json.dump(dicts,dicts_c_t)
##################################Save the data we got##################             
        data_corrupted_analytic_change_CNodes = {'C_N':C_NODES,'Im': Matrix_Im,'Gr':Matrix_Gr,
                                  'Na':Matrix_Na}
############################Save the data we got in this part########################  
        import pickle
        M_Corruption_CNodes = open(File_name + '/' +'save_data_changing_corrupted_CNodes.pkl','wb')
        pickle.dump(data_corrupted_analytic_change_CNodes, M_Corruption_CNodes)
        M_Corruption_CNodes.close()                


#################################################################################################
#################################################################################################        
##################################################################################################


                      
    def Fraction_of_corrupted_Path_number_of_Clusters(self,Itr):#We tend to draw fraction of corrupted path V.S. different value of \tau
        C_clusters = [round(self.N/5),round(2*self.N/5),round(3*self.N/5),round(4*self.N/5)] #put your desired value of \tau in this list, note as we are limmitted in showing the
        #Box plots 7 value of \tau is the maximum allowed
        if self.Diversify ==0:
            Strategy = [6,2,4,1,5]#We have diffrent strategy regarding the relay adversaries:
        else:
            Strategy = [6,2,1,5]
        T  = 0.5 
        c = 0.2
        
        #1:Randomly picked out mix nodes 2:The adversaries owning mix nodes in adistinct location
        #3:-- 4:The adversary farly choses mix nodes from different clusters
        #5:The worst case scenario 
        #6:The best case scenario

        FCPIM = []#FCP stabds for Fraction of Corrupted Path and Im represents Imbalance approach(Low Latency)
        #FCPIM includes FCP of diffrent strtegies whilke we considered Low Latency routing,FCPGR and FCPNA do the same 
        #when we agree to call 'Greedy' as GR and 'Naive' as NA.
        FCPGR = []
        FCPNA = []
        Dictionaries = {} #We save all derived PDFs in this dictioary which comes in useful for the simulation part.
        INDEX = 1 #The dictionary needs index starting from 1.
        for K_ in C_clusters:#We have a loop to exmine changing \tau values
            self.K = K_
            #The three following lists are as same as above lists except for fixing the value of tau for them.
            F_C_P_Im = []
            F_C_P_Gr = []
            F_C_P_Na = [] 
            Dicts = {} #Regarding a fixed \tau we'll have a dictiionary saving the PDFs, IT will add to the Dictionaries
            index_dic = 1#The mentioned dictionary has an index beggining from 1
            for I in range(Itr):#After fixing the value of tau we have to allow for some iterations to make sure we avoide 
                #sampling errors
                F_C_P_Im0 = []#These three lists have simmilar difenitions while we fixed one iteration
                F_C_P_Gr0 = []
                F_C_P_Na0 = []
                C_Nodes   = {}
                Dict_s    = {}#In this case we save PDFs of all strategies in this dictionary
                
                for s in Strategy:
                    if not s==6 and not s==5: #For all strategies except for 5 and 6:
                        self.K = K_
                        D,L,C = self.Corruptions_scenarios(s,c)#We derive D:Distances,L:Latencies and C:Corrupted mix nodes from the implied function
                        self.tau      = T
                        self.Distance = D
                        self.Latency  = L
                        #Now is the time for derin=ving the routing aproaches
                        AA,BB,CC= self.Rout_and_Balance()  #Use this function to fulfill our aims          
                        self.I_Distributions = AA     #Low Latency routing       
                        self.Gready_Balanced_Distributions = BB  #Balanced(Greedy)          
                        self.Naive_Balanced_Distributions = CC   #Balanced(Naive) 
                        #Above distributions are gonna be stored respectively in following dictionaries
                        Mix_Dict ={}
                        Balanced_Greddy_Mixes = {}
                        Balanced_Naive_Mixes = {}
                        counter = 0
                        for j in range( self.L-1):#That's how we save them
                            for i in range(self.W):
                                counter = counter + 1
                                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
                        #Now all the info about the chosen strategy will save in following dictionary                              
                        Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}
                        #As you may remember we have lists for saving the FCP, we may do this regarding FCP function
                        F_C_P_Im0.append(self.FCP(C,Mix_Dict))
                        F_C_P_Gr0.append(self.FCP(C,Balanced_Greddy_Mixes))                   
                        F_C_P_Na0.append(self.FCP(C,Balanced_Naive_Mixes))                       
                    else:
                        if s ==6: #if strategy is s = 6                        
                            D,L,C1 = self.Corruptions_scenarios(s,c) #Here we achieve D:Distances and L:Latencies
                            #From The corruption_Scenarios function and c1 is just a mock value
                            from Corruption import corruptedMix 
                            import numpy as np
                            MM = np.zeros((self.N,3))# This is just a function to intilaize following class,we don't use it later on
                            LM = 0    #Mock parameter        
                            Class_CNs = corruptedMix(MM,c,MM,LM,6) #wE MAINLY USE THIS TO take advantage of fits function for deriving the Cnodes         
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                    #The above codes are loking like whay we explained before just the following are added to attain the
                                    #corrupted mix nodes
                            C_GFP_Im,add1 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Mix_Dict)
                            C_GFP_Gr,add2 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Greddy_Mixes)
                            C_GFP_Na,add3 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Naive_Mixes)

                            F_C_P_Im0.append( self.FCP(C_GFP_Im,Mix_Dict))
                            F_C_P_Gr0.append(self.FCP(C_GFP_Gr,Balanced_Greddy_Mixes))                   
                            F_C_P_Na0.append(self.FCP(C_GFP_Na,Balanced_Naive_Mixes))                    
    
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'corrupted_Mix': C_GFP_Im,'Latency': self.Latency}   
                            Dict_s['Strategy%d'%(s+1)]={'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'corrupted_Mix': C_GFP_Gr,'Latency': self.Latency}                             
                            Dict_s['Strategy%d'%(s+2)]={
                            'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C_GFP_Na,'Latency': self.Latency}                             
                        elif s==5:#For sake of the worst case scenario we're gonna use the following codes which are more or less
                            #equal to what we had already,though we may use different function to achieve CNodes
                            from Clustering import Clustering
                            from MixNetArrangment import Mix_Arrangements
                            from Latency import Latency_and_Distance        
                            from Corruption import corruptedMix                            
                            clusters = Clustering(self.data,self.Clustering,self.K,3,0) 
                            C11 = {}
                            for i in range(self.N):
                                j = i +1
                                C11['PM%d' %j] = False
                            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
                            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function)
                            Distances_ = LatencyDistance.Topology_to_Distance()
                            D = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_) 
                            L = LatencyDistance.Distance_to_Latency()                             
 
                            from Corruption import corruptedMix
                            import numpy as np

                            MM = np.zeros((self.N,3))
                            LM = 0
            
                            Class_CNs = corruptedMix(MM,c,MM,LM,5)
          
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                            C = Class_CNs.Worst_Case(Mix_Dict,False)                            
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}  
                            F_C_P_Im0.append(0)
                            F_C_P_Gr0.append(0)                   
                            F_C_P_Na0.append(0)                            
                            
                            
                Dicts['Dic%d' %index_dic] = Dict_s
                index_dic = index_dic + 1       
                F_C_P_Im.append(F_C_P_Im0)
                F_C_P_Gr.append(F_C_P_Gr0)
                F_C_P_Na.append(F_C_P_Na0)
            Dictionaries['DicT%d' %INDEX] = Dicts
            INDEX = INDEX+1
            M1 = np.matrix(F_C_P_Im) 
            Im = Med((np.transpose(M1)).tolist())
            M2 = np.matrix(F_C_P_Gr)
            Gr = Med((np.transpose(M2)).tolist())            
            M3 = np.matrix(F_C_P_Na)
            Na = Med((np.transpose(M3)).tolist())                
            FCPIM.append(Im)
            FCPGR.append(Gr)
            FCPNA.append(Na)
        Matrix_Im = np.transpose(np.matrix(FCPIM))
        Matrix_Gr = np.transpose(np.matrix(FCPGR))
        Matrix_Na = np.transpose(np.matrix(FCPNA)) 
##################################Save the data for simulations##############################
        import json
        dicts = json.dumps(Dictionaries)
        with open('Dict_save_corrupted_clusters%d.json'%self.Diversify,'w') as dicts_c_t:
            json.dump(dicts,dicts_c_t)
##################################Save the data we got##################             
        data_corrupted_analytic_change_CNodes = {'K':C_clusters,'Im': Matrix_Im,'Gr':Matrix_Gr,
                                  'Na':Matrix_Na}
############################Save the data we got in this part########################  
        import pickle
        M_Corruption_CNodes = open('save_data_changing_corrupted_C_Clusters%d.pkl'%self.Diversify,'wb')
        pickle.dump(data_corrupted_analytic_change_CNodes, M_Corruption_CNodes)
        M_Corruption_CNodes.close()                
##################################################################################################        
        from Plot import PLOT
        Y = []
        if self.Diversify ==0:
            D = ['Best Case Scenario','Random','Fair to clusters','C closest mix nodes','Worst Case Scenario']
        else:
            D = ['Best Case Scenario','Random','C closest mix nodes','Worst Case Scenario']
        D.reverse()
        
        for i in range(len(D)):
            Y.append(Matrix_Im[i,:].tolist()[0])

        
        PLT = PLOT(C_clusters,Y,D,'Number of clusters','Fraction of corrupted paths','C_Clusters_Imbalanced_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Gr[i,:].tolist()[0])
        
        PLT = PLOT(C_clusters,Y,D,'Number of clusters','Fraction of corrupted paths','C_Clusters_Gr_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Na[i,:].tolist()[0])
        
        PLT = PLOT(C_clusters,Y,D,'Number of clusters','Fraction of corrupted paths','C_Clusters_Na_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

#################################################################################################
#################################################################################################        
##################################################################################################


                      
    def Fraction_of_corrupted_Path_methods_of_Clustering(self,Itr,Tau,Limitation):
        if Limitation:
            Methods = ['kmeans','kmedoids']
        else: 
            Methods = ['kmeans','kmedoids','FCM']

        if self.Diversify ==0:
            Strategy = [6,2,4,1,5]#We have diffrent strategy regarding the relay adversaries:
        else:
            Strategy = [6,2,1,5]
        T  = Tau 
        c = 0.2
        
        #1:Randomly picked out mix nodes 2:The adversaries owning mix nodes in adistinct location
        #3:-- 4:The adversary farly choses mix nodes from different clusters
        #5:The worst case scenario 
        #6:The best case scenario

        FCPIM = []#FCP stabds for Fraction of Corrupted Path and Im represents Imbalance approach(Low Latency)
        #FCPIM includes FCP of diffrent strtegies whilke we considered Low Latency routing,FCPGR and FCPNA do the same 
        #when we agree to call 'Greedy' as GR and 'Naive' as NA.
        FCPGR = []
        FCPNA = []
        Dictionaries = {} #We save all derived PDFs in this dictioary which comes in useful for the simulation part.
        INDEX = 1 #The dictionary needs index starting from 1.
        for item in Methods:#We have a loop to exmine changing \tau values
            self.Clustering =  item
            #The three following lists are as same as above lists except for fixing the value of tau for them.
            F_C_P_Im = []
            F_C_P_Gr = []
            F_C_P_Na = [] 
            Dicts = {} #Regarding a fixed \tau we'll have a dictiionary saving the PDFs, IT will add to the Dictionaries
            index_dic = 1#The mentioned dictionary has an index beggining from 1
            for I in range(Itr):#After fixing the value of tau we have to allow for some iterations to make sure we avoide 
                #sampling errors
                F_C_P_Im0 = []#These three lists have simmilar difenitions while we fixed one iteration
                F_C_P_Gr0 = []
                F_C_P_Na0 = []
                C_Nodes   = {}
                Dict_s    = {}#In this case we save PDFs of all strategies in this dictionary
                
                for s in Strategy:
                    if not s==6 and not s==5: #For all strategies except for 5 and 6:

                        D,L,C = self.Corruptions_scenarios(s,c)#We derive D:Distances,L:Latencies and C:Corrupted mix nodes from the implied function
                        self.tau      = T
                        self.Distance = D
                        self.Latency  = L
                        #Now is the time for derin=ving the routing aproaches
                        AA,BB,CC= self.Rout_and_Balance()  #Use this function to fulfill our aims          
                        self.I_Distributions = AA     #Low Latency routing       
                        self.Gready_Balanced_Distributions = BB  #Balanced(Greedy)          
                        self.Naive_Balanced_Distributions = CC   #Balanced(Naive) 
                        #Above distributions are gonna be stored respectively in following dictionaries
                        Mix_Dict ={}
                        Balanced_Greddy_Mixes = {}
                        Balanced_Naive_Mixes = {}
                        counter = 0
                        for j in range( self.L-1):#That's how we save them
                            for i in range(self.W):
                                counter = counter + 1
                                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
                        #Now all the info about the chosen strategy will save in following dictionary                              
                        Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}
                        #As you may remember we have lists for saving the FCP, we may do this regarding FCP function
                        F_C_P_Im0.append(self.FCP(C,Mix_Dict))
                        F_C_P_Gr0.append(self.FCP(C,Balanced_Greddy_Mixes))                   
                        F_C_P_Na0.append(self.FCP(C,Balanced_Naive_Mixes))                       
                    else:
                        if s ==6: #if strategy is s = 6                        
                            D,L,C1 = self.Corruptions_scenarios(s,c) #Here we achieve D:Distances and L:Latencies
                            #From The corruption_Scenarios function and c1 is just a mock value
                            from Corruption import corruptedMix 
                            import numpy as np
                            MM = np.zeros((self.N,3))# This is just a function to intilaize following class,we don't use it later on
                            LM = 0    #Mock parameter        
                            Class_CNs = corruptedMix(MM,c,MM,LM,6) #wE MAINLY USE THIS TO take advantage of fits function for deriving the Cnodes         
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                    #The above codes are loking like whay we explained before just the following are added to attain the
                                    #corrupted mix nodes
                            C_GFP_Im,add1 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Mix_Dict)
                            C_GFP_Gr,add2 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Greddy_Mixes)
                            C_GFP_Na,add3 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Naive_Mixes)

                            F_C_P_Im0.append( self.FCP(C_GFP_Im,Mix_Dict))
                            F_C_P_Gr0.append(self.FCP(C_GFP_Gr,Balanced_Greddy_Mixes))                   
                            F_C_P_Na0.append(self.FCP(C_GFP_Na,Balanced_Naive_Mixes))                    
    
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'corrupted_Mix': C_GFP_Im,'Latency': self.Latency}   
                            Dict_s['Strategy%d'%(s+1)]={'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'corrupted_Mix': C_GFP_Gr,'Latency': self.Latency}                             
                            Dict_s['Strategy%d'%(s+2)]={
                            'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C_GFP_Na,'Latency': self.Latency}                             
                        elif s==5:#For sake of the worst case scenario we're gonna use the following codes which are more or less
                            #equal to what we had already,though we may use different function to achieve CNodes
                            from Clustering import Clustering
                            from MixNetArrangment import Mix_Arrangements
                            from Latency import Latency_and_Distance        
                            from Corruption import corruptedMix                            
                            clusters = Clustering(self.data,self.Clustering,self.K,3,0) 
                            C11 = {}
                            for i in range(self.N):
                                j = i +1
                                C11['PM%d' %j] = False
                            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
                            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function)
                            Distances_ = LatencyDistance.Topology_to_Distance()
                            D = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_) 
                            L = LatencyDistance.Distance_to_Latency()                             
 
                            from Corruption import corruptedMix
                            import numpy as np

                            MM = np.zeros((self.N,3))
                            LM = 0
            
                            Class_CNs = corruptedMix(MM,c,MM,LM,5)
          
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                            C = Class_CNs.Worst_Case(Mix_Dict,False)                            
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}  
                            F_C_P_Im0.append(0)
                            F_C_P_Gr0.append(0)                   
                            F_C_P_Na0.append(0)                            
                            
                            
                Dicts['Dic%d' %index_dic] = Dict_s
                index_dic = index_dic + 1       
                F_C_P_Im.append(F_C_P_Im0)
                F_C_P_Gr.append(F_C_P_Gr0)
                F_C_P_Na.append(F_C_P_Na0)
            Dictionaries['DicT%d' %INDEX] = Dicts
            INDEX = INDEX+1
            M1 = np.matrix(F_C_P_Im) 
            Im = Med((np.transpose(M1)).tolist())
            M2 = np.matrix(F_C_P_Gr)
            Gr = Med((np.transpose(M2)).tolist())            
            M3 = np.matrix(F_C_P_Na)
            Na = Med((np.transpose(M3)).tolist())                
            FCPIM.append(Im)
            FCPGR.append(Gr)
            FCPNA.append(Na)
        Matrix_Im = np.transpose(np.matrix(FCPIM))
        Matrix_Gr = np.transpose(np.matrix(FCPGR))
        Matrix_Na = np.transpose(np.matrix(FCPNA)) 
##################################Save the data for simulations##############################
        import json
        dicts = json.dumps(Dictionaries)
        with open('Dict_save_corrupted_Methods%d.json'%self.Diversify,'w') as dicts_c_t:
            json.dump(dicts,dicts_c_t)
##################################Save the data we got##################             
        data_corrupted_analytic_change_CNodes = {'Methods':Methods,'Im': Matrix_Im,'Gr':Matrix_Gr,
                                  'Na':Matrix_Na}
############################Save the data we got in this part########################  
        import pickle
        M_Corruption_CNodes = open('save_data_changing_corrupted_C_Methods%d.pkl'%self.Diversify,'wb')
        pickle.dump(data_corrupted_analytic_change_CNodes, M_Corruption_CNodes)
        M_Corruption_CNodes.close()                
##################################################################################################        
        from Plot import PLOT
        Y = []
        if self.Diversify ==0:
            D = ['Best Case Scenario','Random','Fair to clusters','C closest mix nodes','Worst Case Scenario']
        else:
            D = ['Best Case Scenario','Random','C closest mix nodes','Worst Case Scenario']
        D.reverse()
        
        for i in range(len(D)):
            Y.append(Matrix_Im[i,:].tolist()[0])

        
        PLT = PLOT(Methods,Y,D,'Clustering','Fraction of corrupted paths','C_Methods_Imbalanced_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Gr[i,:].tolist()[0])
        
        PLT = PLOT(Methods,Y,D,'Clustering','Fraction of corrupted paths','C_Methods_Gr_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Na[i,:].tolist()[0])
        
        PLT = PLOT(Methods,Y,D,'Clustering','Fraction of corrupted paths','C_Methods_Na_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

#######################################################################################################################################################################
#######################################################################################################################################################################
###########################################################################################################################################################################
        R = [0.001, 0.01, 0.1, 1, 10, 100]
        for r in R:
            self.H_parameter = r


    def Fraction_of_corrupted_Path_r(self,Itr):#We tend to draw fraction of corrupted path V.S. different value of \tau
        R = [0.001, 0.01, 0.1, 1, 10, 100] #put your desired value of \tau in this list, note as we are limmitted in showing the
        #Box plots 7 value of \tau is the maximum allowed
        if self.Diversify ==0:
            Strategy = [6,2,4,1,5]#We have diffrent strategy regarding the relay adversaries:
        else:
            Strategy = [6,2,1,5]
        T  = 0.5 
        c = 0.2
        
        #1:Randomly picked out mix nodes 2:The adversaries owning mix nodes in adistinct location
        #3:-- 4:The adversary farly choses mix nodes from different clusters
        #5:The worst case scenario 
        #6:The best case scenario

        FCPIM = []#FCP stabds for Fraction of Corrupted Path and Im represents Imbalance approach(Low Latency)
        #FCPIM includes FCP of diffrent strtegies whilke we considered Low Latency routing,FCPGR and FCPNA do the same 
        #when we agree to call 'Greedy' as GR and 'Naive' as NA.
        FCPGR = []
        FCPNA = []
        Dictionaries = {} #We save all derived PDFs in this dictioary which comes in useful for the simulation part.
        INDEX = 1 #The dictionary needs index starting from 1.
        for r in R:
            self.H_parameter = r
            #The three following lists are as same as above lists except for fixing the value of tau for them.
            F_C_P_Im = []
            F_C_P_Gr = []
            F_C_P_Na = [] 
            Dicts = {} #Regarding a fixed \tau we'll have a dictiionary saving the PDFs, IT will add to the Dictionaries
            index_dic = 1#The mentioned dictionary has an index beggining from 1
            for I in range(Itr):#After fixing the value of tau we have to allow for some iterations to make sure we avoide 
                #sampling errors
                F_C_P_Im0 = []#These three lists have simmilar difenitions while we fixed one iteration
                F_C_P_Gr0 = []
                F_C_P_Na0 = []
                C_Nodes   = {}
                Dict_s    = {}#In this case we save PDFs of all strategies in this dictionary
                
                for s in Strategy:
                    if not s==6 and not s==5: #For all strategies except for 5 and 6:
                        D,L,C = self.Corruptions_scenarios(s,c)#We derive D:Distances,L:Latencies and C:Corrupted mix nodes from the implied function
                        self.tau      = T
                        self.Distance = D
                        self.Latency  = L
                        #Now is the time for derin=ving the routing aproaches
                        AA,BB,CC= self.Rout_and_Balance()  #Use this function to fulfill our aims          
                        self.I_Distributions = AA     #Low Latency routing       
                        self.Gready_Balanced_Distributions = BB  #Balanced(Greedy)          
                        self.Naive_Balanced_Distributions = CC   #Balanced(Naive) 
                        #Above distributions are gonna be stored respectively in following dictionaries
                        Mix_Dict ={}
                        Balanced_Greddy_Mixes = {}
                        Balanced_Naive_Mixes = {}
                        counter = 0
                        for j in range( self.L-1):#That's how we save them
                            for i in range(self.W):
                                counter = counter + 1
                                Mix_Dict['PM%d' %counter] = self.I_Distributions[i,:,j].tolist()
                                Balanced_Greddy_Mixes['PM%d' %counter] = self.Gready_Balanced_Distributions[i,:,j].tolist()
                                Balanced_Naive_Mixes['PM%d' %counter] = self.Naive_Balanced_Distributions[i,:,j].tolist()
                        #Now all the info about the chosen strategy will save in following dictionary                              
                        Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}
                        #As you may remember we have lists for saving the FCP, we may do this regarding FCP function
                        F_C_P_Im0.append(self.FCP(C,Mix_Dict))
                        F_C_P_Gr0.append(self.FCP(C,Balanced_Greddy_Mixes))                   
                        F_C_P_Na0.append(self.FCP(C,Balanced_Naive_Mixes))                       
                    else:
                        if s ==6: #if strategy is s = 6                        
                            D,L,C1 = self.Corruptions_scenarios(s,c) #Here we achieve D:Distances and L:Latencies
                            #From The corruption_Scenarios function and c1 is just a mock value
                            from Corruption import corruptedMix 
                            import numpy as np
                            MM = np.zeros((self.N,3))# This is just a function to intilaize following class,we don't use it later on
                            LM = 0    #Mock parameter        
                            Class_CNs = corruptedMix(MM,c,MM,LM,6) #wE MAINLY USE THIS TO take advantage of fits function for deriving the Cnodes         
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                    #The above codes are loking like whay we explained before just the following are added to attain the
                                    #corrupted mix nodes
                            C_GFP_Im,add1 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Mix_Dict)
                            C_GFP_Gr,add2 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Greddy_Mixes)
                            C_GFP_Na,add3 = Class_CNs.Best_case_chosen_corrupted_mix_nodes(Balanced_Naive_Mixes)

                            F_C_P_Im0.append( self.FCP(C_GFP_Im,Mix_Dict))
                            F_C_P_Gr0.append(self.FCP(C_GFP_Gr,Balanced_Greddy_Mixes))                   
                            F_C_P_Na0.append(self.FCP(C_GFP_Na,Balanced_Naive_Mixes))                    
    
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'corrupted_Mix': C_GFP_Im,'Latency': self.Latency}   
                            Dict_s['Strategy%d'%(s+1)]={'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'corrupted_Mix': C_GFP_Gr,'Latency': self.Latency}                             
                            Dict_s['Strategy%d'%(s+2)]={
                            'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C_GFP_Na,'Latency': self.Latency}                             
                        elif s==5:#For sake of the worst case scenario we're gonna use the following codes which are more or less
                            #equal to what we had already,though we may use different function to achieve CNodes
                            from Clustering import Clustering
                            from MixNetArrangment import Mix_Arrangements
                            from Latency import Latency_and_Distance        
                            from Corruption import corruptedMix                            
                            clusters = Clustering(self.data,self.Clustering,self.K,3,0) 
                            C11 = {}
                            for i in range(self.N):
                                j = i +1
                                C11['PM%d' %j] = False
                            arrangment = Mix_Arrangements(clusters.Mixes,self.Diversify,clusters.Labels,clusters.Centers,C11,self.frc,True)
                            LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function)
                            Distances_ = LatencyDistance.Topology_to_Distance()
                            D = LatencyDistance.Distance_to_real_end_to_end_Latency(Distances_) 
                            L = LatencyDistance.Distance_to_Latency()                             
 
                            from Corruption import corruptedMix
                            import numpy as np

                            MM = np.zeros((self.N,3))
                            LM = 0
            
                            Class_CNs = corruptedMix(MM,c,MM,LM,5)
          
                            self.tau      = T
                            self.Distance = D
                            self.Latency  = L
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
                            C = Class_CNs.Worst_Case(Mix_Dict,False)                            
                            Dict_s['Strategy%d'%s]={'Mix_Dict':Mix_Dict,'Greedy_Balanced_Mix_Dict':Balanced_Greddy_Mixes,
                              'Naive_Balanced_Mix_Dict' : Balanced_Naive_Mixes,'corrupted_Mix': C,'Latency': self.Latency}  
                            F_C_P_Im0.append(0)
                            F_C_P_Gr0.append(0)                   
                            F_C_P_Na0.append(0)                            
                            
                            
                Dicts['Dic%d' %index_dic] = Dict_s
                index_dic = index_dic + 1       
                F_C_P_Im.append(F_C_P_Im0)
                F_C_P_Gr.append(F_C_P_Gr0)
                F_C_P_Na.append(F_C_P_Na0)
            Dictionaries['DicT%d' %INDEX] = Dicts
            INDEX = INDEX+1
            M1 = np.matrix(F_C_P_Im) 
            Im = Med((np.transpose(M1)).tolist())
            M2 = np.matrix(F_C_P_Gr)
            Gr = Med((np.transpose(M2)).tolist())            
            M3 = np.matrix(F_C_P_Na)
            Na = Med((np.transpose(M3)).tolist())                
            FCPIM.append(Im)
            FCPGR.append(Gr)
            FCPNA.append(Na)
        Matrix_Im = np.transpose(np.matrix(FCPIM))
        Matrix_Gr = np.transpose(np.matrix(FCPGR))
        Matrix_Na = np.transpose(np.matrix(FCPNA)) 
##################################Save the data for simulations##############################
        import json
        dicts = json.dumps(Dictionaries)
        with open('Dict_save_corrupted_r%d.json'%self.Diversify,'w') as dicts_c_t:
            json.dump(dicts,dicts_c_t)
##################################Save the data we got##################             
        data_corrupted_analytic_change_CNodes = {'r':R,'Im': Matrix_Im,'Gr':Matrix_Gr,
                                  'Na':Matrix_Na}
############################Save the data we got in this part########################  
        import pickle
        M_Corruption_CNodes = open('save_data_changing_corrupted_C_r%d.pkl'%self.Diversify,'wb')
        pickle.dump(data_corrupted_analytic_change_CNodes, M_Corruption_CNodes)
        M_Corruption_CNodes.close()                
##################################################################################################        
        from Plot import PLOT
        Y = []
        if self.Diversify ==0:
            D = ['Best Case Scenario','Random','Fair to clusters','C closest mix nodes','Worst Case Scenario']
        else:
            D = ['Best Case Scenario','Random','C closest mix nodes','Worst Case Scenario']
        D.reverse()
        
        for i in range(len(D)):
            Y.append(Matrix_Im[i,:].tolist()[0])

        
        PLT = PLOT(R,Y,D,'r','Fraction of corrupted paths','C_R_Imbalanced_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Gr[i,:].tolist()[0])
        
        PLT = PLOT(R,Y,D,'r','Fraction of corrupted paths','C_R_Gr_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

        Y = []
        for i in range(len(D)):
            Y.append(Matrix_Na[i,:].tolist()[0])
        
        PLT = PLOT(R,Y,D,'r','Fraction of corrupted paths','C_R_Na_FCP%d.png'%self.Diversify)
        PLT.scatter_line(True)

#################################################################################################
#################################################################################################        
##################################################################################################


############################################Advanced EXP#############################################
#####################################################################################################
##############################################################################################
##############################################################################################        
###############################################Set3########################################
###########################################Chang r##################################
    def Entropy_Latency_VS_r(self,Iteration,Tau):
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
        R = [ 0.01,0.05, 0.1,0.5, 1]
        for r in R:
            self.H_parameter = r
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

                LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function)

                Distances = LatencyDistance.Topology_to_Distance()

                Latency = LatencyDistance.Distance_to_Latency() 
            
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
        with open('Dict_save_r.json','w') as dicts_file:
            json.dump(dicts,dicts_file) 
##################################Save the data we got##################             
        data_clusters_analytic = {'r':R,'H_Im': H_Imbalanced,'L_Gr':L_Greedy,
                                  'H_Gr':H_Greedy,'L_Na':L_Naive,'H_Na':H_Naive}
        
        data_dic = json.dumps(data_clusters_analytic)
        with open('save_data_changing_r.json','w') as data_clusters:
            json.dump(data_dic,data_clusters)

##############Plot the data##########################################       
            
            
        from Plot import PLOT
        Y = [H_Naive, H_Greedy, H_Imbalanced]
        D = ['Naive Balanced','Greedy Balanced','Low Latency(Imbalance)']
        X_Label = 'r'
        Y_Label = 'Entropy(bit)'
        Name = 'r_Enteropy%d.png'%self.Diversify
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True,True)
        Y = [L_Naive, L_Greedy, L_Imbalanced]
        Y_Label = 'Latency (sec)'
        Name = 'r_Latency%d.png'%self.Diversify
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.scatter_line(True,True)
         

##############################################################################################        
###############################################Set3########################################
###########################################Chang r##################################
    def Trade_off_r(self,Iteration):
        L_T_R = []
        H_T_R = []
        import statistics
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance 
        R = [ 0.01,0.05, 0.1,0.5, 1]        
        for r in R:
            H_Imbalanced = []
            L_Imbalanced = []
  
       
            num_Clusters = []        
            corrupted_Mix = {}
            for k in range(1,self.N+1):
                corrupted_Mix['PM%d' %k] = False
            Dictionaries = {}
            INTRY = 1


            T = [ 0.01, 0.2,0.4,0.6,0.8,1]
            for Tau in T:
            
                self.tau = Tau
                self.OneTime = False                
                
                self.H_parameter = r
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

                    LatencyDistance = Latency_and_Distance(arrangment.Topology,self.Speed_Function)

                    Distances = LatencyDistance.Topology_to_Distance()

                    Latency = LatencyDistance.Distance_to_Latency() 
            
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
            H_T_R.append(H_Imbalanced)
            L_T_R.append(L_Imbalanced)
            
            
 


##############Plot the data##########################################       
            
            
        from Plot import PLOT
        Y = [H_T_R]
        D = ['Low Latency']
        X_Label = 'r'
        Y_Label = 'Entropy(bit)'
        Name = 'tarde_off_Entropy%d.png'%self.Diversify
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)
        Y = [L_T_R]
        Y_Label = 'Latency (sec)'
        Name = 'tarde_off_Latency%d.png'%self.Diversify
        PLT = PLOT(R,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(True)        











 












  
    
    
    
    
    
    