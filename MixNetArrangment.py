# -*- coding: utf-8 -*-
"""
Mix net arrangment: is a class for making determination of mix nodes assignment to the mix net
It contains Vannilla and diversification methods
"""
import numpy as np
import matplotlib.pyplot as plt
class Mix_Arrangements(object):
    
    def __init__(self,Mix_Nodes,Diversify, Labels,Clusters,Corrupted_Nodes,fraction_of_data,OneTime):
        
    
        self.Clustered_Mixes = Mix_Nodes
        self.Frc = fraction_of_data

        self.Diversify = Diversify #True when you want to make the mix nodes diversified
        
        self.num_mixes_per_cluster = Labels
        
        self.Correpted_topology = Corrupted_Nodes
        (a1,b1) = np.shape(self.Clustered_Mixes)
        self.A = a1
        self.C = Clusters
        self.N = round(a1*self.Frc)
        self.W = int(self.N/3)
        (a2,b2) = np.shape(self.C)
        self.K = a2 #number of clusters
        self.List = [] # list of clusters
        for i in range (len(self.num_mixes_per_cluster)):
            self.List.append(i)
            
        self.L = 3
        self.MNodes = []
        for i in range(self.N):
            self.MNodes.append(self.Clustered_Mixes[i,:].tolist())        
        self.Initilization_Value = True # This value eliniate if we need to consider the centriod cluster when we
        #are in scenario 1 of diversification algorithm
        
        self.Topology = self.Arrangment()
        if self.Diversify == 0 and OneTime:
            self.CNs = self.mapping()


     
        
        
    def Arrangment(self):#Arrange the mix net regarding to self.Diversify
        
        if self.Diversify == 0:
            Topology = self.Diversification()
           
            
        elif self.Diversify ==1:
            
            Topology = self.Random_assignment()
            
        elif self.Diversify == 2:
            Topology = self.Random_Worst_Case()
        return Topology
        
        
    def Random_assignment(self):#If you are not eager to diverdify the mix net we
        #Use the random function to assign the mix nodes to the mix network
        Topology = np.zeros((self.L,self.W,3))
        N = self.A
        for i in range(self.L):
            for j in range(self.W):
                index_i = round((N-1)*np.random.rand(1)[0])
                N = N -1
                
                Topology[i,j,:] = self.Clustered_Mixes[index_i,:]# randomly choose a mix 
                #from the set of mix nodes
                
                self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0)
                #delete the chosen mix nodes
                
        return Topology    
        
        
    def Random_Worst_Case(self):
        #We would like to create the worst case scenario happen with random selection
        Topology = np.zeros((self.L,self.W,3))# Here is the approximate scenario

        for i in range(self.L):
            for j in range(self.W):
                
                Topology[i,j,:] = self.Clustered_Mixes[i*(self.W)+j,:]

        return Topology       
        
        
        
        
    def Diversification(self):# Make the diversification with this function
        Topology = np.zeros((self.L,self.W,3)) #Diversified mix net will be 
        #discribed with this matrix

        for i in range(self.L):# here we use the diversification for all three layers
            num_current_clusters = 0
            for item in self.num_mixes_per_cluster:
                
                if item !=0: # Check which scenario we have to use from the diversification algorithm
                    num_current_clusters = num_current_clusters + 1
            
            if self.W < num_current_clusters:
                
                Topology[i,:,:]  = self.div1(self.Initilization_Value)
            else:
                if self.W > num_current_clusters:
                    
                    Topology[i,:,:]  = self.div3()
                else:
                    
                    Topology[i,:,:]  = self.div2()

        return Topology         
                
        
        
        
    def div1(self,Initialization_Median): #Scenario 1 of diversification Algorithm

        current_Layer = np.zeros((self.W,3)) #Diversified layer will be save in this matrix
        
        if Initialization_Median: # if you feel like using the centriod cluster for 
            #selection of first mix node you may follow this function
            Distance = float('inf')
            for I in range(self.K):
                 Center = self.C[I,:]
                 Dist = 0
                 for J in range(self.K):#Centriod is held by the cluster which has 
                     #the minimum distance of the other clusters
                     if (self.num_mixes_per_cluster[J] !=0):
                         Dif = Center - self.C[J,:]
                         Dist = Dist + np.sqrt(np.sum(np.multiply(Dif,Dif)))
                         if Dist < Distance:
                             Distance = Dist
                             index_C = I
        else:   #I f you want to consider the first cluster randomly see below              
            state = True # a variable to make sure we pick up first mix nodes 
            #from the cluster which have atleast one mix nodes
            while(state == True):
                index_C = round((self.K-1)*np.random.rand(1)[0]) #indedx_c is for cluster
            
                if self.num_mixes_per_cluster[index_C] == 0:
                    pass
                else:
                    state = False
        
        index_M = round((self.num_mixes_per_cluster[index_C]-1)*np.random.rand(1)[0])
        #index_M representing the chosen mix node that selected randomly from the derived cluster
        
        self.num_mixes_per_cluster[index_C] = self.num_mixes_per_cluster[index_C] - 1
        #we always need to update number of mix nodes in the chosen ckuster after picking up the mic node
        
        
        
        interval = 0# this helps us to find out the location of mix node in clustered mixes function
        
        for i in range(index_C):
            interval = interval + self.num_mixes_per_cluster[i] #location of that mix nodes
            
        
        index_i = interval + index_M
        
        current_Layer[0,:] = self.Clustered_Mixes[index_i,:] #Relocate the mix node from the matrix
        #to the current layer

        self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0)
        #now remove the row represnting the location of chosen(assigned) mix node
        
        
        
        
        for j1 in range(1,self.W):#Now chose the other mix nodes for the current layer
            Distance_From_Centriod = 0
            
            for j2 in range(self.K):
                Dis = 0
                
                if self.num_mixes_per_cluster[j2] != 0:#chose the next cluster such that it maximizes the distance from the centriod of
                    #of the chosen mix nodes if the cluster had atleast one mix node
                    for j3 in range(j1):
                        dis = current_Layer[j3,:]-self.C[j2,:]#measure the distance
                        Dis = Dis + np.sqrt(np.sum(np.multiply(dis,dis)))  
                    if (Dis > Distance_From_Centriod ):#chose the index_c
                        Distance_From_Centriod = Dis
                        index_C = j2
                    
            index_M = round((self.num_mixes_per_cluster[index_C]-1)*np.random.rand(1)[0])
            #As mentiond randomly chose a mix net and remove it from the matrix as follows
            self.num_mixes_per_cluster[index_C] = self.num_mixes_per_cluster[index_C] - 1
            interval = 0
        
            for i in range(index_C):
                interval = interval + self.num_mixes_per_cluster[i]
            index_i = interval + index_M
        
            current_Layer[j1,:] = self.Clustered_Mixes[index_i,:] 

        
            self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0)
        
            
        return current_Layer    #return the current layer as diversified layer    
  
    
    
    
    
    
    
    
    def div2(self):#Make the scenario 2 materrelize here

        current_Layer = np.zeros((self.W,3))
        j = 0
        for I in range(self.K):
            #first make sure we are considering the clusters with mix nodes available then
            #try to pick one mix nodes randomly from the mix nodes then simillar 
            #to privious function try to add it to current list and make it eliminated from the 
            #the mix clusterd matrix
            if (self.num_mixes_per_cluster[I] !=0):
                index_C = I
                index_M = round((self.num_mixes_per_cluster[index_C]-1)*np.random.rand(1)[0])
        
                self.num_mixes_per_cluster[index_C] = self.num_mixes_per_cluster[index_C] - 1
        
        
        
                interval = 0
        
                for i in range(index_C):
                    interval = interval + self.num_mixes_per_cluster[i]
            
        
                index_i = interval + index_M

        
                current_Layer[j,:] = self.Clustered_Mixes[index_i,:] 
        
                self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0) 
                j = j +1
        return current_Layer 
    
    
    
    
    
    
    
    def div3(self):#Use this function when it comes to scenario3 from diversification
        #Algorithem when self.W >self.nem_cluster

        current_Layer = np.zeros((self.W,3))

        j = 0
        for I in range(self.K):#First pick randomly one mix nodes from the clusters which have 
            #atleast one mix nodes looking like the scenario 2
            if (self.num_mixes_per_cluster[I] !=0):
                index_C = I
                index_M = round((self.num_mixes_per_cluster[index_C]-1)*np.random.rand(1)[0])
        
                self.num_mixes_per_cluster[index_C] = self.num_mixes_per_cluster[index_C] - 1

                interval = 0
        
                for i in range(index_C):
                    interval = interval + self.num_mixes_per_cluster[i]
            
        
                index_i = interval + index_M

        
                current_Layer[j,:] = self.Clustered_Mixes[index_i,:] 
        
                self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0) 
                j = j +1
                

        for k in range(j,self.W): #Now we need to determine other mix nodes withwhich we may complete the
            #current layer
            ProDist = []# we define a probaility distribution regarding the number of 
            # mix nodes in unpdated clusters

            
            for item in self.num_mixes_per_cluster:# make the multinomial distribution here

                ProDist.append(item/sum(self.num_mixes_per_cluster))

                
            Realization_of_Mul_distribution = np.random.multinomial(1, ProDist, size=1)
            #Now make a realization of it to know the next cluster


            index_C = np.sum(np.multiply( np.matrix(self.List),np.matrix(Realization_of_Mul_distribution)))
           #Now the rest of the code is looking like what we discussed already, first select
            #the mis node then make sure it has been asigned to the current layer and remove it from
            #the matrix   

            index_M = round((self.num_mixes_per_cluster[index_C]-1)*np.random.rand(1)[0])
        
            self.num_mixes_per_cluster[index_C] = self.num_mixes_per_cluster[index_C] - 1
        
            interval = 0
        
            for i in range(index_C):
                interval = interval + self.num_mixes_per_cluster[i]
            
        
            index_i = interval + index_M

        
            current_Layer[k,:] = self.Clustered_Mixes[index_i,:] 
        
            self.Clustered_Mixes = np.delete(self.Clustered_Mixes ,index_i,axis = 0) 
            # do this loop as long as we need mix node for the current layer
            
               
        return current_Layer    
    def mapping(self):
        MaP = []
        for I in range(self.N):
            for J in range(self.W):
                for K in range(self.L):
                    if self.MNodes[I][0] == self.Topology[K,J,0] and self.MNodes[I][1] == self.Topology[K,J,1] and self.MNodes[I][2] == self.Topology[K,J,2]:
                        MaP.append(J+self.W*K)
                        
        Map =[]
       
        for ii in range(self.N):
            
            for jj in range(self.N):
                if MaP[jj] == ii:
                    Map.append(jj)                        
                    
        self.Map = Map              
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
        List = []
        for k in range(self.N):
            K = k +1
            if self.Correpted_topology['PM%d' %K] == True:
                List.append(k)
            
            
        for item in List:
            
            for J in range(self.N):
                if self.Map[J] == item:
                    j = J +1
                    
                    CNodes['PM%d' %j] = True
        return CNodes
        
    def Spherical(self,A): # This is a function to transform the cartesian point to Spherical
        #cordinate that is useful for making the clustered mix nodes skeched
        import math
        
        R = math.sqrt(A[0]**2 + A[1]**2 + A[2]**2)

        Phi = math.acos(A[2]/R)
        

        Par1 =  A[1]
        Par2 =  A[0]

        ArcTan_Frac = math.atan(Par1/Par2) # Foe arc tan we just can't bank on what python 
        #give back us, as tan has 2 roots in 2pi arc
        
        if (Par1>=0 and Par2>=0):
            Teta = ArcTan_Frac
        elif(Par1>=0 and Par2<0):
            Teta =  ArcTan_Frac + math.pi
        elif(Par1<0 and Par2>=0):
            Teta =  ArcTan_Frac+ 2*math.pi
        elif(Par1<0 and Par2<0):
            Teta =  math.pi+ArcTan_Frac
            
        if Teta> math.pi:
            Teta = Teta - 2*math.pi
        elif Teta<-math.pi:
            Teta = Teta + 2*math.pi
        return (Teta*180)/math.pi,((Phi)*180)/math.pi
    
    
    
    def Topology_plt(self): #Simillar to clustering here we try to plot the mix net topology
        Data_PLOT = []



        for j in range(self.L):
            lat = []
            lon = []            
            

  
            for i in range(self.W):
                A = self.Topology[j,i,:]

   
                
                Theta,Phi = self.Spherical(A)
                lon.append(Theta)
                lat.append(90-Phi)
            Data_PLOT.append([lat,lon])
            
        plt.close('all') 

        for i in range(len(Data_PLOT)):
            a = Data_PLOT[i][0]
            b = Data_PLOT[i][1]
            font1 = {'family':'Times Roman','color':'blue','size':22}
            plt.scatter(b,a, marker = 'h',linewidths =0.16, alpha=0.5) 
            plt.title('Topology of the mix net',fontdict = font1)
            plt.xlabel("Longtitude")
            plt.ylabel("Latitude")
        plt.savefig('Topology.eps', format='eps') #save the plpoted dots
                    
                    
                    
                    
                    


