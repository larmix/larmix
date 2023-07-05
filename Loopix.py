# -*- coding: utf-8 -*-
"""
Loopix
"""

import numpy  as np

class MixNet(object):
    #In this class we are gonna make an instantiation of the  mix net
    def __init__(self,env, Linkdelays, Mixes,N,Policy,Type):
        self.env = env
        #This is agian simpy  environment
        #This list of delays necessary for computing the end_to_end latency
        self.M = Mixes
        #List of mixes created from the mix class
        self.N = len(self.M) #Number of all mix nodes
        self.W = int(self.N/2)#Number of mix nodes in each mixing layer
        self.LD = Linkdelays
        self.Policy = Policy
        self.Type = Type
        self.LL = []
        #End_to_end latency are added to this list
        self.EN =[]
        # Entropy or distribution are appended to this list for each individual message
    def Message_Traveling(self,message):

        Pro1 = np.random.multinomial(1, [1/self.W]*self.W, size=1).tolist()[0]

         

        M1 = Pro1.index(1)
        Pro2 = np.random.multinomial(1, self.Policy['PM'+str(M1+1)+self.Type], size=1).tolist()[0]
        M2 = Pro2.index(1)
                

        Mixnode1 = self.M[M1]
        message.mtime.append(self.env.now)#The time of start       
        yield self.env.process(Mixnode1.Receive_and_send(message))

        yield self.env.timeout(self.LD['PM'+str(M1+1)+'PM'+str(self.W+M2+1)])
        Mixnode2 = self.M[M2+self.W]
        yield self.env.process(Mixnode2.Receive_and_send(message))
      
        message.mtime.append(self.env.now)#The time of leaving the mix net should be written down
        
        self.LL.append(message.mtime[1]-message.mtime[0])#The latency will be added to the latency list
           
        self.EN.append(message.prob)#The message dist will be added to the entropy list
       
