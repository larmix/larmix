# -*- coding: utf-8 -*-
"""
Generator
"""



import numpy as np
from scipy.stats import expon
from Message_ import message




class Message_Genartion_and_mix_net_processing(object):
    
    def __init__(self,env,Mixes,capacity,MNet,num_target,delay):
        
        self.env = env
        
        self.Mixes = Mixes
        
        self.capacity = capacity
        

        self.MNet = MNet
        
        self.NT = num_target
        
        self.delay = delay




    def Prc(self):
        import math
        #This function is written to be used for generating the messages through the mix network

        ID = 0 #The id of the first target messages
        for i in range(1000):#generate a fix number of messages to initiate the network
            TARGET = False
            target_id = -1
       
            client_id = 'C0'
            M = message('Message%02d' %i,self.NT,target_id,client_id)#message is being created
            self.env.process(self.MNet.Message_Traveling(M))#message send to the mix net
        i = 1000
        while True:# Create other messages by an exponential delay
            t2 = expon.rvs(scale=self.delay)#The exponential delay between two succeeding messages
            yield self.env.timeout(t2)
            

            TARGET = False
            target_id = -1
            if(ID < self.NT):
                y = np.random.multinomial(1, [1/2,1/2], size=1)[0][0]
                if y==1:
                    TARGET = True
                if TARGET:
                    target_id = ID
                    ID = ID + 1

            M = message('Message%02d'%i,self.NT,target_id,client_id)
            self.env.process(self.MNet.Message_Traveling(M) ) 
                
            i = i + 1









