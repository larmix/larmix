
# -*- coding: utf-8 -*-
"""
corrupted Mixes
"""

import numpy as np
import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def nCr(n,r):
    import math
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def Path_Fraction(a,b,c,Dict,W):
    Term = 0
    if len(a) !=0 and len(b) !=0 and len(c) !=0:
        for item1 in a:
            for item2 in b:
                for item3 in c:
                    Term = Term + (1/W)*(Dict['PM%d' %item1][item2-W-1])*(Dict['PM%d' %item2][item3-2*W-1])
    return Term

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
        
class  corruptedMix(object):
    def __init__(self,Data,Budget,Mixes,Labels,strategy):
        (a1,b1) =  np.shape(Mixes)
        self.N = a1
        self.W = int(self.N/3)
        self.Budget = Budget
        self.LL = np.matrix([Labels])
        
        self.CNodes = round(self.N*Budget)
        
        self.Mixes = Mixes
        self.Labels = Labels
        
        self.strategy = strategy
        
        self.data = Data
        
        
        

    def corrupted_mix_nodes(self):
        if self.strategy == 0:
            self.CNs = self.NO_corruption()
        elif self.strategy == 1:
            self.CNs = self.C_random()
        elif self.strategy ==2:
            self.CNs = self.n_closest_mix_nodes()
        elif self.strategy == 3:
            INDX = sort_of_clusters(self.LL.tolist()[0])
            self.CNs = self.C_greedy(INDX)
        elif self.strategy == 4:
            self.CNs = self.fair_to_clusters()
        elif self.strategy == 5:
            self.CNs = self.non_realistic()
        
            
            
            
            
            
    def C_random(self):
        CNodes = {}
        for i in range(self.N):
            
            coin = np.random.multinomial(1, [self.Budget,1-self.Budget], size=1)[0][0]
            if coin == 1:
                    
                j = i +1
                CNodes['PM%d' %j] = True
            else:
                j = i +1
                CNodes['PM%d' %j] = False
        return CNodes
            
            
            
            
    def NO_corruption(self):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
        return CNodes
            
    
    def C_greedy(self,Index):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False

        L = self.Labels
    
        CN = 0
        entry = 0
        while (CN < self.CNodes):
            index = Index[entry]
            i = 0
            if index == 0:
                i = 0
            else:
                for t in range(index ):
                    i = i + L[t]
                
            for k in range(i, i + L[index]):
                
                if (CN < self.CNodes):
                    
                    j = k + 1
                    CNodes['PM%d' %j] = True
                    CN = CN + 1
                else:
                    break
            entry = entry + 1
        return CNodes
                    
                    
    def fair_to_clusters(self):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False

        L = self.Labels
        n = len(L)
        LL = [0]*n
        CN = 0
        while(CN < self.CNodes):
            
            Restricted = True
            
            while(Restricted == True):
            
                r = round((n-1)*np.random.rand(1)[0])
                t =0
            
                if r == 0:
                    pass
                else:
                    for i in range(r):
                        t = t + L[i]
                if  (LL[r] < L[r]):
                    
                    
                    j = t + LL[r] + 1
                    CNodes['PM%d' %j] = True
                    LL[r] = LL[r] + 1
                    CN = CN + 1
                    Restricted = False
            
        return CNodes        
                
            
        

    def Distance_of_two_points(self,point1,point2):
        import math
    
        P1 = point1.tolist()[0]
        P2 = point2.tolist()[0] 
    
        d1 = P1[0]-P2[0]
        d2 = P1[1]-P2[1]    
        d3 = P1[2]-P2[2]   
        Distance = math.sqrt(d1**2+d2**2+d3**2)
        return Distance

    def n_closest_mix_nodes(self):
        import numpy as np
        N, b = np.shape(self.data)
        D = np.zeros((N,N))
    
        for i in range(N):
            for j in range(N):
                if not (i==j):
                         D[i,j] = self.Distance_of_two_points(self.data[i,:], self.data[j,:])
                     
        A = 10000
               
        for i in range(A):
            List = []
            Radius = ((i+1)/A)*np.max(D)
            for j in range(N):   
                counter = 0
                L = []
                for k in range(N):
                    if not D[j,k] > Radius:
                        counter = counter + 1
                        L .append(k)
                        if not  counter < self.CNodes:
                    
                            List = L

                            break
                if not  counter < self.CNodes:
                    

                    break
            if not  counter < self.CNodes:


                break                        
                
                
                    
        
    
    
        corrupted_mix_nodes = np.zeros((self.CNodes,3))

        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
            
        for item in List:
            j = item +1
            CNodes['PM%d' %j] = True
            
        return CNodes
    
    
    def Greedy_For_Fairness(self,Dict):
        SET = []
        for X in range(1,int(self.N/3)+1):
            SET.append(X)
        W_List = findsubsets(set(SET),int(self.CNodes/3))
        import numpy as np
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False        
        C_i = int(self.CNodes/3)

        if nCr(self.W,C_i) < 60*self.N+1:
            import timeit        
            Num_mix_nodes = []
            for i in range(1,self.W+1):
                Num_mix_nodes.append(i)                
            WL = findsubsets(set(Num_mix_nodes), C_i)            
        else:
            import random
            WL = []
            LIs = []
            for j in range(self.W):
                LIs.append( j+1)            
            while len(WL)<60*self.N+1:

                RNDM = tuple(random.sample(LIs, C_i))
                if not (RNDM in WL):
                    WL.append(RNDM)
            WL = set(WL)
                            
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

            
            while(c < self.CNodes):

                Par = -1
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
                A = item
                B = Index_x
                C = Index_y
                
                
                
        for i in range(self.N):
            if not ((i+1) in A) and not ((i-self.W) in B) and not ((i-2*self.W) in C):
                j = i +1
                CNodes['PM%d' %j] = False
            else:
                j = i +1
                CNodes['PM%d' %j] = True                

        return CNodes,Max
        
        
        
        
        
        
        
        
        
        
        

    def Greedy_For_Mix_nodes(self,Dict):
        List = []
        for i in range(1,self.N+1):
            List.append(i)
            

        C_L = findsubsets(set(List),3)
        a_dis = [0]
        b_dis = [0]
        c_dis = [0]
        Max = 0


        for item in(C_L):
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

        for i in range(4,self.CNodes+1):
            Par1 = 0
            for m in range(1,self.W+1):        
                if not (m in a_dis):
                    Parameter1 = Path_Fraction(a_dis+[m],b_dis,c_dis,Dict,self.W)
                    if Parameter1>Par1:
                        Ind1 = m
                        Par1 = Parameter1


            for m in range(self.W+1,2*self.W+1):       
                if not (m in b_dis):
                    Parameter1 = Path_Fraction(a_dis,b_dis+[m],c_dis,Dict,self.W)
                            
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
                                
        CNodes = {}
        for i in range(self.N):
            if not ((i+1) in (a_dis)) and not ((i+1) in (b_dis)) and not ((i+1) in (c_dis)):
                j = i +1
                CNodes['PM%d' %j] = False
            else:
                j = i +1
                CNodes['PM%d' %j] = True  
                
                FCP1 = Path_Fraction(a_dis,b_dis,c_dis,Dict,self.W)
        return CNodes,FCP1
    
    def Best_case_chosen_corrupted_mix_nodes(self,Dict):
        CNodes1, FCP1 = self.Greedy_For_Fairness(Dict)


        return CNodes1, FCP1

   

    def Worst_Case(self,Dict,state):
        if state:           
            if self.CNodes<self.W+1:
                C,M = self.Worst_Case_1(Dict)
            elif self.W < self.CNodes < 2*self.W+1:
                C,M = self.Worst_Case_2(Dict)
            elif self.CNodes > 2*self.W:
                C,M = self.Worst_Case_3(Dict)
        else:
            if self.CNodes<self.W+1:
                C,M = self.Worst_Case_0(Dict)
            elif self.W < self.CNodes < 2*self.W+1:
                C,M = self.Worst_Case_0(Dict)
            elif self.CNodes > 2*self.W:
                C,M = self.Worst_Case_0(Dict)            
        return C,M
    
    def Worst_Case_0(self,Dict):
        import numpy as np
        LAYERS =[1,2,3]
        x = [[],[],[]]

        if self.CNodes < self.W+1:
            Layer_c = 1 +round(2*np.random.rand())
            for i in range(1,self.CNodes+1):
                Next_c = 1 +round((self.W-1)*np.random.rand())
                if not (Next_c in x[Layer_c-1]):
                    x[Layer_c-1].append(Next_c+(Layer_c-1)*self.W)
        elif self.W < self.CNodes < 2*self.W+1:
            Layer_0 = 1 +round(2*np.random.rand())
            Layer_next = Layer_0
            while(Layer_next == Layer_0):
                    Layer_next = 1 +round(2*np.random.rand())
                    
            for i in range(1,self.W+1):                
                x[Layer_0-1].append(i+(Layer_0-1)*self.W) 
            
            for i in range(1,self.CNodes-self.W+1):
                Next_c = 1 +round((self.W-1)*np.random.rand())
                if not (Next_c in x[Layer_next-1]):
                    x[Layer_next-1].append(Next_c+(Layer_next-1)*self.W)                 

        elif self.CNodes>2*self.W:
            Layer_c = 1 +round(2*np.random.rand())
            LAYERS.pop(LAYERS.index(Layer_c))
            for i in range(1,self.CNodes+1-2*self.W):
                Next_c = 1 +round((self.W-1)*np.random.rand())
                if not (Next_c in x[Layer_c-1]):
                    x[Layer_c-1].append(Next_c+(Layer_c-1)*self.W) 
            
            for i in range(1,self.W+1):
                x[LAYERS[0]-1].append(i+(LAYERS[0]-1)*self.W) 
                x[LAYERS[1]-1].append(i+(LAYERS[1]-1)*self.W) 
            
        A = x[0]
        a = A
        B =x[1]
        b = B
        C = x[2]
        c = C
                
            
        Term1 = 0
        if len(a) !=0 and len(b) !=0 and len(c) !=0:
            for item1 in a:
                for item2 in b:
                    for item3 in c:
                        Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        

        CNodes = {}
        for i in range(self.N):
            if not ((i+1) in A) and not ((i+1) in B) and not ((i+1) in C):
                j = i +1
                CNodes['PM%d' %j] = False
            else:
                j = i +1
                CNodes['PM%d' %j] = True                

        return CNodes,Term1            
                
                
            
            
            
            
        
                
                

            
            

    def Worst_Case_1(self,Dict):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
        INDEX = self.CNodes 
        x = []
        for i in range(1,self.W+1):
            x.append(i)       
        C_L = findsubsets(set(x),INDEX)


        Min = 1000000
        for j in range(3):
            
            for itemm in C_L:
                item = list(itemm)
                a = []
                b = []
                c = []
                for mixes in item:
                    if mixes+j*self.W < self.W +1:
                        a.append(mixes+j*self.W)
                    elif self.W< mixes+j*self.W< 2*self.W+1:                   
                        b.append(mixes+j*self.W)
                    elif mixes+j*self.W> 2*self.W:
                        c.append(mixes+j*self.W)

                Term1 = 0
                if len(a) !=0 and len(b) !=0 and len(c) !=0:
                    for item1 in a:
                        for item2 in b:
                            for item3 in c:
                                Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        
                if Min > Term1:
                    Min = Term1
                    A = a
                    B = b
                    C = c 

            for i in range(self.N):
                if not ((i+1) in A) and not ((i+1) in B) and not ((i+1) in C):
                    j = i +1
                    CNodes['PM%d' %j] = False
                else:
                    j = i +1
                    CNodes['PM%d' %j] = True                

        return CNodes,Min



     
    def Worst_Case_2(self,Dict):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
        INDEX = self.CNodes - self.W
            
        x1 = []
        x2 = []
        x3 = []
        for i in range(1,self.W+1):
            x1.append(i)
            x2.append(i+self.W)
            x3.append(i+2*self.W)
        C_L_1 = findsubsets(set(x2+x3),INDEX)
        C_L_2 = findsubsets(set(x1+x3),INDEX)                            
        C_L_3 = findsubsets(set(x1+x2),INDEX)

        Min = 1000000
        for itemm in C_L_1:
            item = list(itemm)
            a = x1
            b = []
            c = []
            for mixes in item:
                if mixes < 2*self.W +1:
                    b.append(mixes)
                else:                   
                     c.append(mixes)

            Term1 = 0
            if len(a) !=0 and len(b) !=0 and len(c) !=0:
                for item1 in a:
                    for item2 in b:
                        for item3 in c:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        
            if Min > Term1:
                Min = Term1
                A = a
                B = b
                C = c  

        for itemm in C_L_2:
            item = list(itemm)
            a = []
            b = x2
            c = []
            for mixes in item:
                if mixes < self.W +1:
                    a.append(mixes)
                else:                   
                     c.append(mixes)

            Term1 = 0
            if len(a) !=0 and len(b) !=0 and len(c) !=0:
                for item1 in a:
                    for item2 in b:
                        for item3 in c:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        
            if Min > Term1:
                Min = Term1
                A = a
                B = b
                C = c 

        for itemm in C_L_3:
            item = list(itemm)
            a = []
            b = []
            c = x3
            for mixes in item:
                if mixes < self.W +1:
                    a.append(mixes)
                else:                    
                     b.append(mixes)

            Term1 = 0
            if len(a) !=0 and len(b) !=0 and len(c) !=0:
                for item1 in a:
                    for item2 in b:
                        for item3 in c:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        
            if Min > Term1:
                Min = Term1
                A = a
                B = b
                C = c 

            for i in range(self.N):
                if not ((i+1) in A) and not ((i+1) in B) and not ((i+1) in C):
                    j = i +1
                    CNodes['PM%d' %j] = False
                else:
                    j = i +1
                    CNodes['PM%d' %j] = True                

        return CNodes,Min










        
    
    def Worst_Case_3(self,Dict):
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
            

        if self.W > 1:
            INDEX = self.CNodes - 2*self.W
            
            List = []
            x2 = []
            x3 = []
            for i in range(1,self.W+1):
                List.append(i)
                x2.append(i+self.W)
                x3.append(i+2*self.W)
                
            

            C_L = findsubsets(set(List),INDEX) 
            Min = 1000000
            for itemm in C_L:
                item = list(itemm)
                a = item
                b = x2
                c = x3
                Term1 = 0
                if len(a) !=0 and len(b) !=0 and len(c) !=0:
                    for item1 in a:
                        for item2 in b:
                            for item3 in c:
                                    Term1 = Term1 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-2*self.W-1]) 
                                        
                if Min > Term1:
                    Min = Term1
                    A = a
                    B = b
                    C = c                

                a = List
                b = item
                c = x3
                Term2 = 0
                if len(a) !=0 and len(b) !=0 and len(c) !=0:
                    for item1 in a:
                        for item2 in b:
                            for item3 in c:
                                    Term2 = Term2 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-1])*(Dict['PM%d' %int(item2+self.W)][int(item3)-2*self.W-1])            
        
                if Min > Term2:
                    Min = Term2
                    A = a
                    B = []
                    for term in b:
                        B.append(term +self.W)
                    C = c    
    
                a = List
                b = x2
                c = item
                Term3 = 0
                if len(a) !=0 and len(b) !=0 and len(c) !=0:
                    for item1 in a:
                        for item2 in b:
                            for item3 in c:
                                    Term3 = Term3 + (1/self.W)*(Dict['PM%d' %int(item1)][int(item2)-self.W-1])*(Dict['PM%d' %int(item2)][int(item3)-1])    
    
                if Min > Term3:
                    Min = Term3
                    A = a
                    B = b
                    C = []
                    for term in c:
                        C.append(term +2*self.W)
            
            for i in range(self.N):
                if not ((i+1) in A) and not ((i+1) in B) and not ((i+1) in C):
                    j = i +1
                    CNodes['PM%d' %j] = False
                else:
                    j = i +1
                    CNodes['PM%d' %j] = True                

        return CNodes,Min    
    
    
    
    
    
    
    
    def non_realistic(self,Algorithm):
        Wc = round(self.CNodes/3)
        CNodes = {}
        for j in range(3):
            C_N = 1
            for i in range(round(self.N/3)):
                if not C_N > Wc:
                    
                
                    CNodes['PM%d' %(i+j*round(self.N/3))] = True
                    C_N = C_N + 1
                else:
                    CNodes['PM%d' %(i+j*round(self.N/3))] = False
 
        return CNodes

        

            
