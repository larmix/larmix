# -*- coding: utf-8 -*-
"""
Main File: In this file we evaluate LARMix
"""


            
            

from LARMIX import LARMIX
C = LARMIX()

#Figure2 : To make the figure 2 in the paper you can use the following code
C.Formula1()


#Figure 4a and Figure 5, 

#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.Basic_Exp_Diversification('Fig4_a',128*3,1)           
    
# Figure 4b 
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations  
del C
C = LARMIX()      
C.Basic_Exp_Random('Fig4_b',128*3,2)              
#Figure 4c   
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations   
del C
C = LARMIX()    
C.Basic_Exp_WC('Fig4_c',128*3,2)    

#Table3
#The function takes three arguments: the first argument specifies the maximum end to end delay
# The second one is used to specify the name of the experiment, 
#the third argument determines the number of mix nodes (which should be a multiple of 3), 
#The last one specifies the number of iteration
del C
C = LARMIX()
C.Table3(0.2,'Table3',128*3,5) 
         
#Fig6
#The function takes two arguments: the first argument specifies list of  maximum end to end delays
# The second one is used to specify the name of the experiment, 
#the third argument determines the number of mix nodes (which should be a multiple of 3), 
#The last one specifies the number of iteration
del C
C = LARMIX()
C.Maximum_tau_mu('Fig6',[0.2,0.25,0.3,0.35,0.4,0.45],128*3,1)    
#Fig7

#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 and 10), 
#and the last argument represents the number of iterations
del C
C = LARMIX() 
C.Network_Size('Fig7',300,2)   
#Fig8 and Fig10
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 ), 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.FCP_Greedy('Fig8and10',32*3,2)
#Fig9
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.FCP_Cnodes('Fig9',32*3,2)        
            
#Fig11
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 ), 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.Claps('Fig11',32*3,2) 

             
#Table 4
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes in each layer, 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.Two_Layers_VS_LARMIX('Loopix_Larmix',128,1)  

#Client_Latency
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes , 
#and the last argument represents the number of iterations 
del C
C = LARMIX()
C.Client_Latency('Client_Latency',20,1)





