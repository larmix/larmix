# -*- coding: utf-8 -*-
"""
Mix_Node
"""

from scipy.stats import expon
#Mix class would be necessary for making instantiation of mixes for the mix net,
 #here we assume continuous mix which flushes the input messages with an exponential
# delay
class Mix(object):
    def __init__(self,env,name,capacity,corrupt,num_target,delay):
       # A mix node instantiation may have different attributes we described them
        #as follows.
        self.env =env    
        #1)environment: which simply passes the environment of discrete event simulation,
        self.name = name
        #2)name: which is associated with the name of mixes starting from 00 to N-1,
        self.capacity = capacity
        #3)capacity: which regards the maximum number of messages that a mix can keep in its pool.
         # In many cases, we may consider it as infinite number but it may cause an increase  in
        #latency if mixes have limited capacity so it's better to consider it as a simpy environment
        #(Actually it can motivate why we're going to make mix net balanced)
        self.corrupt = corrupt
       #4)corrupt: showing whether a mix has been compromised by the adversary or not that is worthful
        #for computing the probability of the mix.
        self.num_target = num_target
        self.delay = delay
        self.tarP = [0]*self.num_target

        #6)Target probability actually refers to the probability of mix node which is computed by
    #summing over all messages distributions it should be zero at the first time as it has no message.
        self.pool = 0
        #The number of messages currently stays in the mix node
   
    def Receive_and_send(self,message):
        #This function is for receiving messages, making them processed by mix nodes,
        #and flushing them out after an exponential delay.

        with self.capacity.request() as req:
            yield req
            #As we told a  mix node has a capacity. Thereby it will check if it can receive
            #an extra message, if so it  allows the message to come, otherwise  it should wait
            #until a message is flushed out of the pool. so yield req make the message wait for that
            self.pool = self.pool + 1
            # Upon leting a message in number of messages in the pool will be added by one.
           
            #If the mix node is not corrupted the probability of the message and the mix
            #should be updated.(The probability of the message will be added to the mix node)
            if not self.corrupt:                
                for j in range(self.num_target):
                    self.tarP[j] = self.tarP[j] + message.prob[j]
                   
               

           
           # The message will lie in the mix node for a random exponential time
            t1 = expon.rvs(scale=self.delay)
            yield self.env.timeout(t1)
            #Simpy will yield for this amount of time
            if not self.corrupt:
                #If the mix node is not corrupted the probability of the message and the mix node
                #should be updated before forwarding the message to the next hop. Actually,
                #the probability of the message will be changed to the probability of the mix
                #divided by the number of messages existing in the pool, later on,
                #this probability is substracted from the mix probability to make it update.
               
                for j in range(self.num_target):
                    message.prob[j] = self.tarP[j]/self.pool
                    self.tarP[j] = self.tarP[j] - message.prob[j]
                   
            #The number of messages will be reduced
            self.pool = self.pool - 1
           
