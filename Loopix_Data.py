# -*- coding: utf-8 -*-
"""
Data
"""



"""
Datasets is a class for making a data set or receive one which will be adopted to
what we expect as a matrix to have the mix nodes locations
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self,dataset,Number_of_MixNodes,Goal):
        self.G = Goal
        self.dataset = dataset
        self.N = Number_of_MixNodes
        self.P = math.pi
        self.R = 6378.137 #Radius of earth
        self.Data = self.data_set() # make your costumized data set
        self.Plt = self.plt_data() # plot the mix nodes around the earth
        
        
        self.Dta = 0
        
        
    
    def Gen_dataset(self):# A function for genrating random dataset
        # first generat random Phi and theta then move them to Cartesian coordinate system
    
        Phi = np.matrix(np.random.uniform(0,360,self.N))

        Theta = np.matrix(np.random.uniform(0,360,self.N))

        R_E = self.R

        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
   
        return Mix_Location
    
    def extentention_of_data(self):# When we need more mix nodes than the dataset's
        Data = pd.read_csv(self.dataset)
        if self.dataset=='worldcities.csv':
            Data.rename(columns = {'lng':'lon'}, inplace = True)
        data = Data
        Latitude = data['lat']   #Phi
        Longitude = data['lon']  #Teta
        Length = len(data)
        i = Length
        M_Latitude = np.matrix(Latitude)
        
        M_Longtitude = np.matrix(Longitude)
        
        while(self.N > i): # Make the data coppied to be prepared for the self.N 
            if (self.N >= (i + Length)):
                M_Latitude = np.concatenate((M_Latitude,np.matrix(Latitude)), axis=1)
                M_Longtitude = np.concatenate((M_Longtitude,np.matrix(Longitude)), axis=1)
                i = i + Length
            else:
                n1 = self.N - i
                M_Latitude = np.concatenate((M_Latitude,np.matrix(Latitude.head(n = n1))), axis=1)
                M_Longtitude = np.concatenate((M_Longtitude,np.matrix(Longitude.head(n = n1))), axis=1)
                i = self.N
                
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R

        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
   
        return Mix_Location
    def RIPE(self):
        import json
        import numpy as np 
        
        with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 

            data = json.load(json_file)
            I_Key = []
            Lon =   []
            Lat =   []
            Mix_nodes = []
            
            
        for i in range(self.N):
            item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            while (item) in Mix_nodes:
                item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            Mix_nodes.append(item)
            I_Key.append(data[item]['i_key'])
            Lon.append(float(data[item]['longitude']))            
            Lat.append(float(data[item]['latitude'])) 
        NYM_Data = {'lat':Lat, 'lon':Lon, 'i_key':I_Key}
        Latitude = NYM_Data['lat']   #Phi
        Longitude = NYM_Data['lon']  #Teta
        self.Dta = {'lat':Lat, 'lon':Lon}
        M_Latitude = np.matrix(Latitude)
        M_Longtitude = np.matrix(Longitude)
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R
        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
        x = Mix_Location[0,:].tolist()[0]
        y = Mix_Location[1,:].tolist()[0]
        z = Mix_Location[2,:].tolist()[0]
        
        NYM_D = { 'x':x , 'y':y , 'z':z,'i_key':I_Key}

        dics = json.dumps(NYM_D)
        with open(self.G + 'RIPE.json','w') as df_ripe:
            json.dump(dics,df_ripe)        
                
        
        Mix_Locations_Transposed = np.transpose(Mix_Location)
        
        return Mix_Locations_Transposed        
    def NYM(self):
        import json
        import numpy as np
        with open('latency_old_with_location_cleaned_up_new.json') as json_file: 

            data = json.load(json_file)
            I_Key = []
            Lon =   []
            Lat =   []
            Mix_nodes = []
            
            
        for i in range(self.N):
            item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            while (item) in Mix_nodes:
                item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            Mix_nodes.append(item)
            I_Key.append(data[item]['i_key'])
            Lon.append(float(data[item]['longitude']))            
            Lat.append(float(data[item]['latitude'])) 
        NYM_Data = {'lat':Lat, 'lon':Lon, 'i_key':I_Key}
        Latitude = NYM_Data['lat']   #Phi
        Longitude = NYM_Data['lon']  #Teta
        self.Dta = {'lat':Lat, 'lon':Lon}
        M_Latitude = np.matrix(Latitude)
        M_Longtitude = np.matrix(Longitude)
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R
        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
        x = Mix_Location[0,:].tolist()[0]
        y = Mix_Location[1,:].tolist()[0]
        z = Mix_Location[2,:].tolist()[0]
        
        NYM_D = { 'x':x , 'y':y , 'z':z,'i_key':I_Key}

        dics = json.dumps(NYM_D)
        with open('NYM.json','w') as df_nym:
            json.dump(dics,df_nym)        
                
        
        Mix_Locations_Transposed = np.transpose(Mix_Location)
        
        return Mix_Locations_Transposed
    
    def data_set(self): 
        
        if self.dataset == 'NYM':
            Mix_Locs = self.NYM()
        elif self.dataset == 'RIPE':
            
            Mix_Locs = self.RIPE()
            
        else:
            Mix_Locs = self.data_set_()
        
        return Mix_Locs

    def data_set_(self):# make the data
        if self.dataset == 'random.csv':
            
            return self.Gen_dataset()
   
        Data = pd.read_csv(self.dataset)
        if self.dataset=='worldcities.csv':
            Data.rename(columns = {'lng':'lon'}, inplace = True)
        if len(Data)<self.N:
            return self.extentention_of_data()
        data = Data.sample(n = self.N)
        Latitude = data['lat']   #Phi
        Longitude = data['lon']  #Teta
        self.Dta = data[['lat','lon']]
        M_Latitude = np.matrix(Latitude)
        
        M_Longtitude = np.matrix(Longitude)
        
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R

        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
        
        Mix_Locations_Transposed = np.transpose(Mix_Location)
   
        return Mix_Locations_Transposed
        


    def plt_data(self):#Here we plot the selected mix nodes on earth 2d map
        import plotly.express as px
