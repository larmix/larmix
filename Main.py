# -*- coding: utf-8 -*-
"""
Main File: In this file we evaluate LARMix
"""

from LARMIX import LARMIX
C = LARMIX()

#Figure1 : To make the figure one in the paper you can use the following code
C.Formula1()

#Figure 4a and Figure 5, 
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations 
C.Basic_Exp_Diversification('Fig4_a',128*3,1)           
    
# Figure 4b 
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations        
C.Basic_Exp_Random('Fig4_b',128*3,2)   

#Figure 4c   
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations       
C.Basic_Exp_WC('Fig4_c',128*3,2)    

#Table3
#The function takes three arguments: the first argument specifies the maximum end to end delay
# The second one is used to specify the name of the experiment, 
#the third argument determines the number of mix nodes (which should be a multiple of 3), 
C.Table3(0.2,'Table3',32*3) 
         
#Fig6
#The function takes two arguments: the first argument specifies list of  maximum end to end delays
# The second one is used to specify the name of the experiment, 
#the third argument determines the number of mix nodes (which should be a multiple of 3), 
C.Maximum_tau_mu('Fig6',[0.25,0.3,0.4],32*3)   

#Fig7
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 and 10), 
#and the last argument represents the number of iterations 
C.Network_Size('Fig7',300,2)   

#Fig8 and Fig10
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 ), 
#and the last argument represents the number of iterations 
C.FCP_Greedy('Fig8and10',32*3,2)

#Fig9
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3), 
#and the last argument represents the number of iterations 
C.FCP_Cnodes('Fig9',32*3,2)        
            
#Fig11
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes (which should be a multiple of 3 ), 
#and the last argument represents the number of iterations 
C.Claps('Fig11',32*3,2) 
         
#Table 4
#The function takes three arguments: the first argument is used to specify the name of the experiment, 
#the second argument determines the number of mix nodes in each layer, 
#and the last argument represents the number of iterations 
C.Two_Layers_VS_LARMIX('Loopix_Larmix',128,2)  