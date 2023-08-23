# -*- coding: utf-8 -*-
"""
LARMIX
"""

from Datasets import Dataset

from Bridge import Bridge


from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle

from Message import message

from Mix_Node import Mix

from Mix_net import MixNet

from Message_Genartion_and_mix_net_processing import Message_Genartion_and_mix_net_processing
import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def Ent(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_
def Refine(List,D):
    List1 = []
    for item in List:
        List1.append(round((10**D)*item)/(10**D))
    return List1
      
        
import numpy as np

import pickle
  
from Simulation import Simulation
class LARMIX(object):
    
    def __init__(self):
        self.data = 'RIPE'        
        self.Dimention = 16*3 #A mixnet with 3 layers each of which has 16 nodes
        self.Clustering = 'kmedoids'
        self.Routing = True
        self.Balancing = True
        self.N = self.Dimention        
        self.num_Clusters = 5        
        self.r = 1       #A fixed parameter in formula 1 in the paper
        self.Tau = 0.001     #Randomness in formula 1  
        self.Algorithm = 'Greedy'
        self.Diversify = 0        #Having the mixnodes diversified when this parameter is 0, if it's 1 or 2 the mixnet is arranged in random or bad case respectively
        self.speed_Function = 'Verloc'  # It's needed for specific cases when we use verloc data     
        self.Decimal_precision = 5      #How accurate our balancing should be  
        self.Layers = 3       
        self.mu = 0.05    #Mixing delay   
        self.Lambda = 0.0001 #10000 messages per seconds enter the mixnet       
        self.Capacity = 10000000000000000000000000000000000000000000000000000000000000000        
        self.H_N = round(self.N/3)  #Hyper parameter to tune the number of messages in simulation      
        self.rate = 100        #Hyper parameter to tune the number of messages in simulation
        self.num_targets = 20       #Number of target messages in simulation 
        self.Iterations = 1        
        self.run_time = 0.32        
        self.NYM = False# We don't want to use NYM dataset
        self.RIPE = True        
        self.strategy = 0      #Adversary strategy which is initialy equale to 0 
        self.frac = 1   #Fraction of used nodes out of available ones
        #In the following we want to make sure we have Figures and Tables folders to store our final results
        import os
        if not os.path.exists('Figures'):
            os.mkdir(os.path.join('', 'Figures'))   
        if not os.path.exists('Tables'):
            os.mkdir(os.path.join('', 'Tables'))             
    def Formula1(self):
        from Formula import test
                
        
        LL = []   
        LLL = []    
        for k1 in range(128):       
            import json
            import numpy as np
            with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 
                data_ = json.load(json_file)      
                
            data1 = data_[0]
            Latency = []
            Mix_nodes = []
            for i in range(128):
                Index = int(500*np.random.rand(1)[0])+1
                while Index in Mix_nodes:
                    Index = int(500*np.random.rand(1)[0])+1
                Mix_nodes.append(Index)
                I_key1 = data_[Index]['i_key']
                Latency.append(int(data1['latency_measurements'][str(I_key1)]))
            LLL.append(Latency)        
        LL.append(Med(LLL))         
        T = [0.01,0.05,0.1,0.15, 0.2,.25,.35,.35,0.4,.45,.5,.55,0.6,.65,.7,.75,0.8,.85,.9,.95,1]        
        r = self.r
        Latency = LL[0]
        
        Test = test(0.5,1)
        Distances = []
        
        import math
        EE = [math.exp(-1)]
        for e in EE:
            
            distance = []
            for item2 in T:
                Dist = Test.InitialDist(Latency,item2,r,e)
                distance.append(7-Test.kUkullback_leibler(Dist))
            Distances.append(distance)
                
        from Plot import PLOT
        
        P = PLOT(T,Distances,['Ripe dataset'],r' $\tau$','Entropy (bit)','Figures/Fig1.png')
        
        
        
        P.scatter_line(True,False)


        
        
    def Basic_Exp_Diversification(self,Name,N,Iteration_):        
            Goal = Name
            DF  = Dataset(self.data,N,Goal)
            Data_set =  DF.data_set()
            Diversify = 0
            C = Bridge(Data_set, self.Clustering, self.Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1,Goal,self.NYM,self.RIPE)
            Iterations2 = Iteration_
            C.Entropy_Latency_VS_Tau(Iterations2)
            del C   
            Sim = Simulation(self.num_targets,Iterations2,self.Capacity,self.run_time,self.mu,self.Lambda,self.H_N,N,self.rate,self.Diversify,self.Clustering,0,Goal)       
            Sim.Entropy_Latency_VS_Tau()
            del Sim
            
            Name = 'Diversification_Basic_EXP' + Goal
            
##################################################Plot#######################################
            PATH =  Name
            from Plot import PLOT
            import pandas as pd
            Data = pd.read_csv(PATH+'/Analytic.csv', skipinitialspace= True)
            Name_L = '/Fig4_a_Latency' + '.png'            
            Name_E = '/Fig4_a_Entropy' + '.png'
            A = [Data['Latancy(N)'], Data['Latancy(Gr)'],   Data['Latancy(Im)']]            
            B = [Data['Anonymity(N)'], Data['Anonymity(Gr)'],   Data['Anonymity(Im)']]
            PLT1 = PLOT(Data['Tau'],A,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r' Randomness ($\tau$)','Latency (sec)','Figures/'+ Name_L)            
            PLT2 =  PLOT(Data['Tau'],B,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r'Randomness ($\tau$)','Entropy (bit)','Figures/'+ Name_E)
            PLT2.scatter_line(True)            
            PLT1.scatter_line(True)
            Name_json = '/Sim.json'
            import json
            with open(PATH + Name_json ,'r') as df:
                Data1 = json.loads(json.load(df))  
            T = Data1['Tau']      
            Y =[Data1['Entropy(Im)'],Data1['Entropy(Gr)'],Data1['Entropy(Na)']] 
            YY =   [Data1['Latency(Im)'],Data1['Latency(Gr)'],Data1['Latency(Na)']] 
            PLTT = PLOT(T,Y,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)','Figures/' + 'Fig5_Entropy.png')    
            PLTT.Box_Plot(True)   
            PLTT2 = PLOT(T,YY,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Latency (sec)','Figures/' + 'Fig5_Latency.png')    
            PLTT2.Box_Plot(True)   
    
    def Basic_Exp_Random(self,Name,N,Iteration_):        
            Goal = Name
            DF  = Dataset(self.data,N,Goal)
            Data_set =  DF.data_set()
            Diversify = 1
            C = Bridge(Data_set, self.Clustering, Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1,Goal,self.NYM,self.RIPE)
            Iterations2 = 1
            C.Entropy_Latency_VS_Tau(Iterations2)
            del C   
            Name = 'Random_Basic_EXP' + Goal
            
##################################################Plot#######################################
            PATH =  Name
            from Plot import PLOT
            import pandas as pd
            Data = pd.read_csv(PATH+'/Analytic.csv', skipinitialspace= True)
            Name_L = '/Fig4_b_Latency' + '.png'            
            Name_E = '/Fig4_b_Entropy' + '.png'
            A = [Data['Latancy(N)'], Data['Latancy(Gr)'],   Data['Latancy(Im)']]            
            B = [Data['Anonymity(N)'], Data['Anonymity(Gr)'],   Data['Anonymity(Im)']]
            PLT1 = PLOT(Data['Tau'],A,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r' Randomness ($\tau$)','Latency (sec)','Figures/'+ Name_L)            
            PLT2 =  PLOT(Data['Tau'],B,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r'Randomness ($\tau$)','Entropy (bit)','Figures/'+ Name_E)
            PLT2.scatter_line(True)            
            PLT1.scatter_line(True)

    def Basic_Exp_WC(self,Name,N,Iteration_):        
            Goal = Name
            DF  = Dataset(self.data,N,Goal)
            Data_set =  DF.data_set()
            Diversify = 2
            C = Bridge(Data_set, self.Clustering, Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1,Goal,self.NYM,self.RIPE)
            Iterations2 = 1
            C.Entropy_Latency_VS_Tau(Iterations2)
            del C   
            Name = 'WC_Basic_EXP' + Goal
            
##################################################Plot#######################################
            PATH =  Name
            from Plot import PLOT
            import pandas as pd
            Data = pd.read_csv(PATH+'/Analytic.csv', skipinitialspace= True)
            Name_L = '/Fig4_c_Latency' + '.png'            
            Name_E = '/Fig4_c_Entropy' + '.png'
            A = [Data['Latancy(N)'], Data['Latancy(Gr)'],   Data['Latancy(Im)']]            
            B = [Data['Anonymity(N)'], Data['Anonymity(Gr)'],   Data['Anonymity(Im)']]
            PLT1 = PLOT(Data['Tau'],A,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r' Randomness ($\tau$)','Latency (sec)','Figures/'+ Name_L)            
            PLT2 =  PLOT(Data['Tau'],B,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r'Randomness ($\tau$)','Entropy (bit)','Figures/'+ Name_E)
            PLT2.scatter_line(True)            
            PLT1.scatter_line(True)
            
    def Table3(self,Max_Latency,Name,N):                                
        from Datasets import Dataset
        Limitation = Max_Latency               
        Goal = Name + str(round(Limitation*1000)) + 'ms'
        data = 'RIPE'
        DF  = Dataset(self.data,N,Goal)
        Data_set =  DF.data_set()
        from Optimum_tau_mu import Detailed_Analysis
        run_time = Limitation+0.1       
        Iterations = 1
        D = Detailed_Analysis(Data_set, self.Clustering, self.Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1
                              ,self.num_targets,Iterations,self.Capacity,run_time,self.Lambda,self.H_N,self.rate,self.Clustering,Goal,self.NYM,self.RIPE)
        D.Entropy_Latency_Analytic(1,Limitation)
        D.Entropy_Latency_Simulation(Limitation)
        import pickle
        delay = int(1000*Limitation)
        Path = 'Maximum_Mixing_Delays' + str(delay)
        with open(Path + '/' +'Delay_dict.pkl', 'rb') as f:
                    Delay_dict = pickle.load(f)         
        Data_Table = {}
        Data_Table['Tau'] = Delay_dict['tau']
        Link_delays = []
        counter = 0
        for item in Delay_dict['mg']:
            Link_delays.append(1000*int(1000*(Limitation-3*item))/1000)
            Data_Table['Link_Delay'] = Link_delays
            Delay_dict['mg'][counter] = 1000*(int(1000*Delay_dict['mg'][counter])/1000)
            counter = counter+1
        Data_Table['Mixing_delay'] = Refine(Delay_dict['mg'],3)                       
        Data_Table['Entropy_Analytic'] = Refine(Delay_dict['hg'],3)   
        Data_Table['Entropy_Simulation'] = Refine(Med(Delay_dict['HG']),3)
        from tabulate import tabulate
        from fpdf import FPDF
        # Convert the dictionary to a list of rows
        table_data = [[key] + value for key, value in Data_Table.items()]
        # Create the table in tabular format
        table = tabulate(table_data, tablefmt='grid')
        # Print the table
        print(table)                
        # Save the table as a PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Dictionary Table', ln=True)
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, table)
        pdf.output('Tables/'+'Maximum_Mixing_Delays' + str(delay)+'.pdf')
        
    def Maximum_tau_mu(self,Name,Delay_List,N):
        Maximum_tau = []
        Maximum_mu  = []
        for m_delay in Delay_List:       
            from Datasets import Dataset
            Limitation = m_delay               
            Goal = str(round(Limitation*1000)) + 'ms'
            data = 'RIPE'
            DF  = Dataset(self.data,N,Goal)
            Data_set =  DF.data_set()                       
            from Optimum_tau_mu import Detailed_Analysis
            run_time = Limitation+0.1            
            Iterations = 1
            D = Detailed_Analysis(Data_set, self.Clustering, self.Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1
                                  ,self.num_targets,Iterations,self.Capacity,run_time,self.Lambda,self.H_N,self.rate,self.Clustering,Goal,self.NYM,self.RIPE)
            D.Entropy_Latency_Analytic(1,Limitation)
            D.Entropy_Latency_Simulation(Limitation)        
            import pickle
            delay = int(1000*Limitation)
            Path = 'Maximum_Mixing_Delays' + str(delay)
            with open(Path + '/' +'Delay_dict.pkl', 'rb') as f:
                        Delay_dict = pickle.load(f) 
            Index = Med(Delay_dict['HG']).index(max(Med(Delay_dict['HG'])))
            Maximum_tau.append(Delay_dict['tau'][Index])
            Maximum_mu.append(1000*(int(1000*Delay_dict['mg'][Index])/1000))
        from Plot import PLOT
        Y = [Maximum_tau,Maximum_mu]
        PLT2 = PLOT(Delay_List,Y,[r'Maximum $\tau$','Maximum mixing delay'],'Maximum E2E delay',[r'Randomness ($\tau$)','Mixing delay'], 'Figures/Fig6.png')
        PLT2.scatter2(True)

    def Network_Size(self,Name,N,Iteration_):
        Diversify = 0
        Dimention = N
        N = Dimention
        
        Goal = 'Network_Size'+Name
        from Datasets import Dataset

        DF  = Dataset(self.data,Dimention,Goal)
        Data_set =  DF.data_set()          
        C = Bridge(Data_set, self.Clustering,Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu, 0,1,Goal,self.NYM,self.RIPE)

        Iteration2 = Iteration_
        tau = 0.6
        C.NetworkSize(Iteration2,tau)
        tau = 0.9
        C.NetworkSize(Iteration2,tau)
        tau = 0.1
        C.NetworkSize(Iteration2,tau)        
        del C
        Path1 = 'NetworkSize'+str(0.1) 
        
        Path2 = 'NetworkSize'+str(0.6)
        
        Path3 = 'NetworkSize'+str(0.9) 
        
        name = '/save_data_changing_nodes.json'
        
        
        
        import json
        
        with open(Path1 + name,'r') as df1:
            Data1 = json.loads(json.load(df1))
            
        with open(Path2 + name,'r') as df2:
            Data2 = json.loads(json.load(df2))
        
        with open(Path3 + name,'r') as df3:
            Data3 = json.loads(json.load(df3))
        
        
        
        
        L = [Data3['L_Gr'],Data2['L_Gr'],Data1['L_Gr']]
        for i in range(3):
            j = 0
            for item in L[i]:
                L[i][j] = L[i][j] 
                j = j + 1
        
        H = [Data3['H_Gr'],Data2['H_Gr'],Data1['H_Gr']]

        
        X = []
        
        for i in range(5):
            X.append(round(N*(0.2+0.2*i)))
        
        
        
        
        from Plot import PLOT
        PLTN = PLOT(X,[H,L],[r'Entropy ($\tau = 0.9$)',r'Entropy ($\tau = 0.6$)',r'Entropy ($\tau = 0.1$)',r'Latency ($\tau = 0.9$)',r'Latency ($\tau = 0.6$)',r'Latency ($\tau = 0.1$)'],'Mixnet size (N)',[r'Entropy $\mathsf{H}(T)}$ (bit)',r'Latency $\bar{l}_{mix}$ (sec)'], 'Figures/Fig7.png')
        
        PLTN.colors = ['blue','red','green','blue','red','green']
        PLTN.markers = ['H','D','v','H','D','v']
        PLTN.Line_style = ['-','-','-',':',':',':'] 
        
        PLTN.scatter3(True)                   


    def FCP_Greedy(self,Name,N,Iteration_):
        Diversify = 0
        Dimention = N
        N = Dimention
        
        Goal = 'FCP_Greedy'+Name
        from Datasets import Dataset

        DF  = Dataset(self.data,Dimention,Goal)
        Data_set =  DF.data_set()  
        from Bridge2 import Adversarial_Bridge
        C = Adversarial_Bridge(Data_set, self.Clustering, self.Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, self.mu,0,1,Goal,self.NYM,self.RIPE)  
        
        
        Iterations = 2*Iteration_
        Iterations2 = Iteration_
        
        C.Fraction_of_corrupted_Path(Iterations2)
        del C
        
        from Simulation import Simulation
        for S in [2]:
            Sim = Simulation(self.num_targets,Iterations,self.Capacity,self.run_time,self.mu,self.Lambda,self.H_N,N,self.rate,self.Diversify,self.Clustering,0,Goal)      
            Sim.Adverserial_VS_Tau(S)
            del Sim





        Path = 'Diversification_FCP' + '/' + 'save_data_changing_corrupted_tau.pkl'
        import pickle
        import numpy as np  
        from Plot import PLOT
        with open(Path, 'rb') as f:
                    D_dict = pickle.load(f)
    
        Tau = D_dict['tau']
        
        Gr = D_dict['Gr']
        Gr_BC = Gr[0,:].tolist()[0]
        
        Gr_CA = Gr[1,:].tolist()[0]
        
        Gr_FC = Gr[2,:].tolist()[0]
        Gr_FC1 = Gr[3,:].tolist()[0]
        
        
        GR = [Gr_BC, Gr_CA, Gr_FC,Gr_FC1]
        
        
        Gr_plt = PLOT(Tau,GR,[  'Worst Case','Single location', 'Diverse location','Random'],r'Randomness ($\tau$)','FCP','Figures/Fig8.png')
        

        Gr_plt.colors = ['navy','red','green','black','orange','cyan']
        Gr_plt.scatter_line(True)
        PTH = 'Diversification_FCP'+ '/' +'2Sim_Corrupted_tau.json'
        import json        
        with open(PTH ,'r') as df:
            Data_Sim1 = json.loads(json.load(df))
            
        Entropy = [Data_Sim1['Entropy(Im)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Na)']]
        Tau = Data_Sim1['Tau']

        PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)', 'Figures/Fig10.png')
        PLT.Box_Plot(True)
        
    def FCP_Cnodes(self,Name,N,Iteration_):
        Diversify = 0
        Dimention = N
        N = Dimention        
        Goal = 'FCP_CNodes' + Name
        from Datasets import Dataset
        DF  = Dataset(self.data,Dimention,Goal)
        Data_set =  DF.data_set()  
        from Bridge2 import Adversarial_Bridge
        C = Adversarial_Bridge(Data_set, self.Clustering, self.Diversify, self.Routing,self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers,self.mu,0,1,Goal,self.NYM,self.RIPE)
        Iterations2 = Iteration_
        C.Fraction_of_corrupted_Path_number_of_Cnodes(Iterations2)
        del C


        Path = 'Diversification_FCP_CNodes' + '/' + 'save_data_changing_corrupted_CNodes.pkl'
        import pickle
        import numpy as np
        
        with open(Path, 'rb') as f:
                    D_dict = pickle.load(f)
        
        from Plot import PLOT
        CN = [0.1,0.2,0.3]
        
        
        Im = D_dict['Im']
        Im_BC = Im[0,:].tolist()[0]
        
        Im_CA = Im[1,:].tolist()[0]
        
        Im_FC = Im[2,:].tolist()[0]
        
        Im_R = Im[3,:].tolist()[0]
        
        Im_WC = Im[4,:].tolist()[0]
        
        IM = [Im_BC, Im_CA, Im_FC, Im_R]
        
        
        Im_plt = PLOT(CN,IM,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP','Figures/Fig9_a.png')
        
        Im_plt.colors = ['navy','red','green','black','orange','cyan']
        Im_plt.scatter_line(True)
        


        #######
        
        
        
        Gr = D_dict['Gr']
        Gr_BC = Gr[0,:].tolist()[0]
        
        Gr_CA = Gr[1,:].tolist()[0]
        
        Gr_FC = Gr[2,:].tolist()[0]
        
        Gr_R =  Gr[3,:].tolist()[0]
        
        Gr_WC = Gr[4,:].tolist()[0]
        
        GR = [Gr_BC, Gr_CA, Gr_FC, Gr_R]
        
        
        Gr_plt = PLOT(CN,GR,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP', 'Figures/Fig9_b.png')
        
        Gr_plt.colors = ['navy','red','green','black','orange','cyan']
        Gr_plt.scatter_line(True)
        
        #######
        
        
        
        NA = D_dict['Na']
        Na_BC = NA[0,:].tolist()[0]
        
        Na_CA = NA[1,:].tolist()[0]
        
        Na_FC = NA[2,:].tolist()[0]
        
        Na_R =  NA[3,:].tolist()[0]
        
        Na_WC = NA[4,:].tolist()[0]
        
        NA = [Na_BC, Na_CA, Na_FC, Na_R]
        
        
        Na_plt = PLOT(CN,NA,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP', 'Figures/Fig9_c.png')
        
        Na_plt.colors = ['navy','red','green','black','orange','cyan']
        Na_plt.scatter_line(True)      
        
        
    def Claps(self,Name,N,Iteration_):
        from CLAPS import LARMix_VS_CLAPS
        from Clustering import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Latency import Latency_and_Distance
        
                
        from Corruption import corruptedMix    
            
            
            
        from Datasets import Dataset
        
        from Bridge import Bridge
        
        from scipy.stats import expon
        import simpy
        import random
        import numpy  as np
        import pickle
        
        from Message import message
        
        from Mix_Node import Mix
        
        from Mix_net import MixNet        
        data = 'RIPE'
        Dimention = N
                
        Goal = Name
        DF  = Dataset(data,Dimention,Goal)
        
        
        Data_set =  DF.data_set()
        
        
        DF.plt_data()
        
        
        
        C_M = 'kmedoids'
        Routing = True
        Balancing = True

        num_Clusters = 2
        
        r = 1
        
        Tau = 0.001
        
        Algorithm = 'Greedy'
        
        
        Diversify = 0
        
        speed_Function = 'Verloc'
        
        Decimal_precision = 5
        
        Layers = 3
        
        mu = 0.02
        
        Lambda = 0.0001
        
        Capacity = 10000000000000000000000000000000000000000000000000000000000000000
        
        H_N = round(N/3)
        
        rate = 15
        
        num_targets = 200
        
        Iterations = Iteration_
        
        run_time = 0.32
        
        NYM = False
        RIPE = True
        
        strategy = 0
        
        frac = 1    
            
            
            
        T = [0,0.2,0.4,0.6,0.8,1] 
        
        Th = [1.25,2.5,5]
        
        L__0 = []
        L__1 = []
        L__2 = []
        E__0 = []
        T__0 = []
            
        E__1 = []
        T__1 = []
        E__2 = []
        T__2 = []
        
        ITERATION = 1
        for Itr in range(ITERATION):    
            
        
            clusters = Clustering(Data_set,C_M,num_Clusters,3,0)
        
            clusters.Data_plt()
            Class_CNs = corruptedMix(Data_set,0.3,clusters.Mixes,clusters.Labels,strategy)
            Class_CNs.corrupted_mix_nodes()
            Corruption = Class_CNs.CNs        
            arrangment = Mix_Arrangements(clusters.Mixes,Diversify,clusters.Labels,clusters.Centers,Corruption,frac,True)
                
            arrangment.Topology_plt()
                    
            Corruption_New = arrangment.mapping()
                
            LatencyDistance = Latency_and_Distance(arrangment.Topology,speed_Function,Goal,NYM,RIPE)
        
            Distances = LatencyDistance.Topology_to_Distance()
            Latency = LatencyDistance.RIPE_Latency
                            
        
            latency = [[],[],[]] 
            entropy = [[],[],[]]  
            time    = [[],[],[]] 
            
            I = 0
                      
        
            for Tau in T:
                
                Claps = LARMix_VS_CLAPS(Latency,Distances,True, Tau,1)
                
        
                E,L = Claps.CLAPS_(1,True)
                I = I+1
                latency[2].append(L)
                entropy[2].append(E)
        
                H_L = Claps.LARMix()
            
                latency[0].append(H_L['L_Greedy'])
                latency[1].append(H_L['L_Naive'])    
            
                entropy[0].append(H_L['H_Greedy'])
                entropy[1].append(H_L['H_Naive']) 
                time[2].append(Claps.CLAPS_time)
                time[1].append(Claps.LARMIX_time[1])    
                time[0].append(Claps.LARMIX_time[0])  
                
            T__0.append(time[0])
            L__0.append(latency[0])    
            E__0.append(entropy[0])    
            T__1.append(time[1])
            L__1.append(latency[1])    
            E__1.append(entropy[1])    
            T__2.append(time[2])
            L__2.append(latency[2])    
            E__2.append(entropy[2])    
            
        import numpy as np    
            
        t_0 = Med(np.transpose(np.matrix(T__0)).tolist())
        t_1 = Med(np.transpose(np.matrix(T__1)).tolist())
        t_2 = Med(np.transpose(np.matrix(T__2)).tolist())    
        
        e_0 = Med(np.transpose(np.matrix(E__0)).tolist())
        e_1 = Med(np.transpose(np.matrix(E__1)).tolist())
        e_2 = Med(np.transpose(np.matrix(E__2)).tolist())
        
        
        l_0 = Med(np.transpose(np.matrix(L__0)).tolist())
        l_1 = Med(np.transpose(np.matrix(L__1)).tolist())
        l_2 =Med(np.transpose(np.matrix(L__2)).tolist())
        
                
        
        File_name = Name + str(N)
        
        Times = [t_0,t_1,t_2]
        
        Entropies = [e_0,e_1,e_2]
        
        Latencies = [l_0,l_1,l_2]
        
        
        Data_CLAPS = {}
        
        Data_CLAPS['Time'] = Times
        Data_CLAPS['Entropy'] = Entropies
        Data_CLAPS['Latency'] = Latencies
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name)) 
        
        import json
        dicts = json.dumps(Data_CLAPS)
        with open(File_name + '/' + 'Data_CLAPS.json','w') as dicts_file:
            json.dump(dicts,dicts_file)

        from Plot import PLOT
        Tau = [0,0.2,0.4,0.6,0.8,1]
        Frac1 = [0]*len(Tau)
        Frac2 =[0]*len(Tau)
        
        Frac3 = [0]*len(Tau)
      
        for i in range(len(Tau)):
            Frac1[i] = Data_CLAPS['Entropy'][0][i]/Data_CLAPS['Latency'][0][i]
            Frac2[i] = Data_CLAPS['Entropy'][1][i]/Data_CLAPS['Latency'][1][i]
            Frac3[i] = Data_CLAPS['Entropy'][2][i]/Data_CLAPS['Latency'][2][i]   
             
        Frac = [Frac1,Frac2,Frac3]        

        PLT4 = PLOT(Tau,Frac,['Greedy','CLAPS','Naive'],r' Randomness ($\tau$)','Entropy/Latency','Figures/Fig11.png')

        PLT4.colors[0] = 'red'
        PLT4.colors[1] = 'green'
        PLT4.colors[2] = 'royalblue'
        PLT4.scatter_line(True)

    def Two_Layers_VS_LARMIX(self,Name,W,Iterations):
        from twoLayers import Loopix_two_layers

        N= W*2
        Mix_delay = 0.075
        mu = 2*(Mix_delay)/3
        Poisson_delay = 0.0001
        Num_Target = 200
        running = 0.6      
        
        tau = 1
        
        Dict = Loopix_two_layers(N,'RIPE','Loopix'+str(N),Iterations,Mix_delay,Poisson_delay,running,Num_Target,tau)
        Entropy_2Layers =  Dict['H_S_Gr'][0]

        L_S_2Layers = Dict['L_S_Gr'][0]
        L_A_2Layers = Dict['L_A_Gr'][0]
            
        Goal = Name
        N = 3*W
        DF  = Dataset(self.data,N,Goal)
        Data_set =  DF.data_set()
        Diversify = 0
        C = Bridge(Data_set, self.Clustering, self.Diversify, self.Routing, self.Balancing, N, self.num_Clusters, self.speed_Function,self.Tau, self.r, self.Decimal_precision, self.Algorithm,self.Layers, mu,0,1,Goal,self.NYM,self.RIPE)
        Iterations2 = Iterations
        C.Entropy_Latency_VS_Tau(Iterations2)
        del C   
        Sim = Simulation(self.num_targets,Iterations2,self.Capacity,self.run_time,mu,self.Lambda,self.H_N,N,self.rate,self.Diversify,self.Clustering,0,Goal)       
        Sim.Entropy_Latency_VS_Tau()
        del Sim
        
        Name = 'Diversification_Basic_EXP' + Goal
        
        PATH =  Name
        from Plot import PLOT
        import pandas as pd
        Data = pd.read_csv(PATH+'/Analytic.csv', skipinitialspace= True)
        
        L_A_3Layers = Data['Latancy(Gr)'][3]

        Name_json = '/Sim.json'
        import json
        with open(PATH + Name_json ,'r') as df:
            Data1 = json.loads(json.load(df))      
        Entropy_3Layers =Med(Data1['Entropy(Gr)'])[3]
        L_S_3Layers = Med(Data1['Latency(Gr)'])[3]
        
        from tabulate import tabulate
        
        # Define the data for the table
        data = [['Analytical Latency',str(L_A_2Layers),str(L_A_3Layers)],
                ['Simulation Latency', str(L_S_2Layers), str(L_S_3Layers)],
                ['Simulation Anonymity', str(Entropy_2Layers), str(Entropy_3Layers)]]
        
        # Print the table
        print(tabulate(data, headers=['Parameter', '2-Layer random routing', '3-Layer LARMix'], tablefmt='fancy_grid'))


        from fpdf import FPDF
        # Create a PDF object
        pdf = FPDF()
        
        # Add a page
        pdf.add_page()
        
        # Set font and font size
        pdf.set_font('Arial', size=12)
        
        # Set table header
        pdf.cell(40, 10, 'Parameter', 1)
        pdf.cell(40, 10, '2-Layer random routing', 1)
        pdf.cell(40, 10, '3-Layer LARMix', 1)
        pdf.ln()
        
        # Set table data
        for row in data:
            for col in row:
                pdf.cell(60, 20, str(col), 1)
            pdf.ln()
        
        # Save the PDF file
        pdf.output('Tables/Loopix_LARMix.pdf')

    def Client_Latency(self,Name,N,Iteration):
        from Client_Latency import Client_Distance_Network




        Frac = 0.4
        Goal = 'Client_average_latency'
        C = Client_Distance_Network(Frac,N,Name)
        
        print(C.Ave_Distance(Iteration))        





      
   
     































