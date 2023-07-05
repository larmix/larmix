
"""
Clustering: is our first step after getting the data set ready for analysis
In a nutshell that makes the computation process relaxed while we are arranging the mix net.
We provide the clustering methods with three options of FCM Kmeans and Kmediods
"""


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class Clustering(object):
    
    def __init__(self,Mix_Locations,Clustering,K,Layers,corrupted_Mix):
        self.M = Mix_Locations
        (a1,b1) = np.shape(self.M)
        
        self.N =a1
        self.W = round(self.N/3)
        self.num_Layers = Layers

        self.corrupted_Mix = corrupted_Mix
        
        self.Clstr = Clustering #Should be mentioned the clustering method :{kmeans,kmediods,FCM,None}

        self.num_clusters = K
        self.Centers,self.Mixes,self.Labels,self.Fisher, self.Map = self.Clusters()
        # Center, Mix nodes after clustering, number of mix nodes per clusters and Fisher metric
        
        
    def No_clustering(self):# Means we don't want to cluster the mix nodes 
        #In other words self.K = self.N and self.Centers = self.Mixes
        Mix_num_per_clusters = []
        
        for i in range(self.N):
            
            Mix_num_per_clusters.append(1) # In each cluster we have one sample it itself is a cluster 
            
        Centers = self.M
        
        Mixes_clustred = self.M
        
        B_D_W_D = float('inf') # as w_distance = 0
        
        return Centers , Mixes_clustred,Mix_num_per_clusters ,B_D_W_D
    
    
        
        
    

    def Clusters(self):# Make the clusters
        if self.Clstr == 'FCM':
            return self.extract_FCM_data()
        elif self.Clstr == 'None':
            return self.No_clustering()
        elif self.Clstr == 'kmeans':
            return self.Clusters_1_2()
        elif self.Clstr == 'kmedoids':
            return self.Clusters_1_2()
         
    def Clusters_1_2 (self): #Here we care about Kmeans and Kmediods
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn_extra.cluster import KMedoids
       
        
        if self.Clstr =='kmeans':
           
            C = KMeans(n_clusters = self.num_clusters).fit(self.M)
        else:
            if self.Clstr == 'kmedoids':
                C = KMedoids(n_clusters=self.num_clusters, random_state=0).fit(self.M)

        M = self.M
        Centers = C.cluster_centers_
        Labels = C.labels_
        F = Labels # Labels is the identity of the cluster to which the mix node i belong
        B_D=0# Between class distance initaied like this
        W_D=0# Within class distance initaied like this
        Mixes_clustred = np.zeros((self.N,3)) # We have a spceial form for considering mix nodes after clustering
        #we put mix nodes of cluster number 0 to K-1 respectively from firstrow to N row
        Mix_labels = np.zeros(self.N) #Simply concerns the number of mix nodes per clusters like
       # [2,3] means 2 mix nodes in cluster 0 and 3 mix nodes in cluster 1
        
        for v in range(self.N):
            for vv in range(self.N):#Try to measure the B_D and W_D
                d = np.abs(M[v,:] - M[vv,:])
                DD = np.multiply(d,d)
                D = np.sqrt(np.sum(DD))
                d1 = Labels[v]
                d2 = Labels[vv]
                if d1==d2:
                    W_D = W_D + D
                else:
                    B_D = B_D + D
                       
        entry = 0 
        Map = []
        for i in range(self.num_clusters):#Here we create the Mix_num_per_clusterd list which helps
            #not lose info of clustering when we feel like arrange the mix nodes in step2

            for j in range(self.N):
                if (F[j]==0):
                    Mixes_clustred[entry , :]= M[j,:]
                    Map.append(j)
                    Mix_labels[entry] = i
                    entry  = entry + 1
            F = F -1
        Mix_num_per_clusters = [0]*self.num_clusters   
        for it in  range(len(Mix_labels)) :
            
            a2 = round(Mix_labels[it])
            
            Mix_num_per_clusters[a2] = Mix_num_per_clusters[a2] + 1
            
            
                       
        return Centers , Mixes_clustred,Mix_num_per_clusters ,B_D/W_D,Map
    
    
    
    def make_the_best_clustering(self):#Sometime the network designer has no idea of the proper 
        #self.K so we defined this function with which they can extraxt the best value of self.k
        #The main idea of finding this value is coming from maximizing the Fisher metric
        Max = 0
        for numC in range(2,self.N):
            
            self.num_clusters = numC
            if self.Clstr =='FCM':
                a,b,c,d = self.extract_FCM_data
            else:
                
            
                a,b,c,d = self.Clusters()
            
            if d> Max:
                Max = d
                
                Index = numC
        return Index
    
    def FCM(self,Eps,m):#We had to creat FCM methode by ourselves:(
        #m is the FCM kind ususally we consider it to be equal to 2 or 3, EPS is a small number like 0.0001
        
        U = np.zeros((self.N,self.num_clusters))#U is a matrix including the membership distributions whicyh is initilized with uniform distributions
        U_Plus = (1/self.num_clusters)*np.ones((self.N,self.num_clusters))#This is U+ means the next update for the membership matrix
        C = np.zeros((self.num_clusters,3)) #C stands for the center of clusters which are in cartesian cordinate
        
        #for i in range(self.num_clusters):#we need to initilize the U matrix , it's done wit some determinestic 
            #distributions like [0,0,0,0,1](If need this uncomment)
            #U[i,i] = 1
        #for i in range(self.num_clusters,self.N): # make clusters update
            #U[i,:] = np.random.multinomial(1, [1/self.num_clusters]*self.num_clusters, size=1)
            
            
        Convergence = False # convergence here means U+ should be equal to U with Eps accuracy
        while not Convergence:# make the loop ticking over unless we get Convergence True
            #meaning we could make U matrix converged(ao does C)
            
            for J in range(self.num_clusters):
                Par1 = 0
                Par2 = 0
                for I in range(self.N):
                    Par1 = Par1 + (U[I,J]**m)*self.M[I,:]
                    Par2 = Par2 + U[I,J]**m
                
                
                C[J,:] = Par1/Par2
            for i in range(self.N):
                for j in range(self.num_clusters):
                    Par3 = 0
                    par3 = np.sum((np.abs(self.M[i,:]-C[j,:])))
                    Par4 = par3**(2/(m-1))
                    for k in range(self.num_clusters):
                        a = np.sum(np.abs(self.M[i,:]-C[k,:]))
                        Par3 = Par3 + Par4/(np.sum(a**(2/(m-1))))
                    
                
                    U_Plus[i,j] = 1/Par3
                
                
            Check_Point = np.max(np.abs(U-U_Plus)) # check point is for checking the convergence criteria
            U = U_Plus
            if Check_Point < Eps:
                
                Convergence = True

        return U,C
    
    def extract_FCM_data(self): #As you may remember we had considered specific list and matrix for shoeing the result of
        #clustered mixed in previous clusters, we're doing the same for FCM in this function
        U,  C  =  self.FCM(0.01,2)

        Clustered_Mix = np.zeros((self.N,3))#This is excatly what we expect of self.Mixes
        Realization = np.zeros((self.N,self.num_clusters))# As in FCM desoite of kmeans or kemdioed we have U matrix
        #We need to make some relizations to see what is the relized label for a certain mix node so we do that here
        Labels = []
        L_num_of_cluster = [] #This is being used for making the number of mix nodes in each clusters clarified
        for j in range(self.num_clusters):
            Labels.append(0)
        Labels2 = np.zeros((1,self.num_clusters))
        for i in range(self.N):
            Realization [i,:] = np.random.multinomial(1, U[i,:], size=1)[0]

        for i in range (self.N):
            for j in range(self.num_clusters):
                if int(Realization[i,j]) == 1:
                    Labels[j] = Labels[j] + 1
                    L_num_of_cluster.append(j+1)#We create it in this loop
        Map = []
        for i in range (self.N):
            for j in range(self.num_clusters):
                if int(Realization[i,j]) == 1:
                    
                    factor = 0
                    for k in range(j):
                        factor = factor + Labels[k]
                    Clustered_Mix[int(factor+Labels2[0,j]),:] = self.M[i,:]
                    Map.append(int(factor+Labels2[0,j]))
                    Labels2[0,j] = Labels2[0,j]+1

        Centers = C 
        
        Mixes_clustred = Clustered_Mix
        
        Mix_num_per_clusters = Labels
        
        B_D=0
        W_D=0
        for v in range(self.N):#Here agian with the samescenario we compute the Fisher metric
   
            for vv in range(self.N):
                d = np.abs(self.M[v,:] - self.M[vv,:])
                DD = np.multiply(d,d)
                D = np.sqrt(np.sum(DD))
                d1 = L_num_of_cluster[v]
                d2 = L_num_of_cluster[vv]
                if d1==d2:
                    W_D = W_D + D
                else:
                    B_D = B_D + D
        Map1 =[]
       
        for i in range(self.N):
            
            for j in range(self.N):
                if Map[j] == i:
                    Map1.append(j)
           

        return Centers , Mixes_clustred,Mix_num_per_clusters ,B_D/W_D,Map1
    
    def mapping(self):
        
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False
        List = []
        for k in range(self.N):
            K = k +1
            if self.corrupted_Mix['PM%d' %K] == True:
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
        if R ==0:
            R = 0.01

        Phi = math.acos(A[2]/R)

        

        Par1 =  A[1]
        Par2 =  A[0]
        if Par2 ==0:
            Par2 = 0.01

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
        
        
        
        
        
    
    def Data_plt(self):#Want to plot clusters mix nodes + clusters with these codes
        Data_PLOT = []

        lat = []
        lon = []
        for i in range(self.num_clusters):# First plot the center of mix nodes
            #considering lon and lat
            
            if self.Clstr == 'None':
                A = self.Centers[i,:].tolist()[0]
            else:
                A = self.Centers[i,:].tolist()
            Theta,Phi = self.Spherical(A)
            lon.append(Theta)
            lat.append(90-Phi)

        Data_PLOT.append([lat,lon])


        for j in range(self.num_clusters):#Now plot mix nodes of cluster
            #from cluster 0 to cluster k-1
            lat = []
            lon = []            
            

            I = 0
            for k in range(j):
                I = I + self.Labels[k]    
  
            for i in range(self.Labels[j]):
                if self.Clstr == 'None':
                    A = self.Mixes[I+i,:].tolist()[0]
                else:
                    A = self.Mixes[I+i,:].tolist()

   
                
                Theta,Phi = self.Spherical(A)
                lon.append(Theta)
                lat.append(90-Phi)
            Data_PLOT.append([lat,lon])
        plt.close('all') #Make the plot shown

        for i in range(len(Data_PLOT)):
            a = Data_PLOT[i][0]
            b = Data_PLOT[i][1]
    
            if i==0:

                plt.scatter(b,a, c = 'b',linewidths =4)
            else:
        
                plt.scatter(b,a, marker = 'h',linewidths =0.16) 
        font1 = {'family':'Times Roman','color':'red','size':22}
        
        plt.title('Clustered mix nodes',fontdict = font1)
        plt.xlabel("Longtitude")
        plt.ylabel("Latitude")
        plt.savefig('Clusters.eps', format='eps') #save the plpoted dots
        
        
            




        
        
