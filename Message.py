# -*- coding: utf-8 -*-
"""
Messages:)
"""


class message(object):
   
    def __init__(self,number, num_target,target_id):
        #This class is about messages, a message may have different attributes
        #considered useful for creating the simulation. The first attribute is
        #the message number that can be referred to as message-id(e.g first
        #message is Message00), then being the target or not can be saved
        #as a value in target_id attribute, a number of target messages and
        #target id( Which target message is this message if it is a target) will
        #be put in the other attributes.

        self.number = number
        self.NT = num_target
        self.target_id = target_id
        self.prob = self.Pro() #Probability of being target messages
        self.mtime = [] # end-to-end latency of the message will store here
        self.Ttime = [] #end to end latency of target messages
       
        #This function is of vital importance for initialing the probability of
        #being a target message, as instance assume we have two target messages
        #and 10 messages in total, in this case for the first target messages
        #this probability list is like [1 0], which means this message is target
        #1 with the probability of 1 and target 2 with probability 0, similarly
        #for the target 2 is [0 1] and for the others should be [0 0].
    def Pro(self):
        P = []

        for i in range(self.NT):  
            P.append(0)
        if  self.target_id!=-1:#For messages which are not a target
            #target id is -1 for the others it starts from 0 to num_target-1
           
            P[self.target_id ] =1
        return P
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    