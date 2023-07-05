# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:31:09 2023

@author: Mahdi
"""


from Plot import PLOT

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

import pandas as pd
import json

Name = 'C:/Users/Mahdi/OneDrive/Desktop/Results/LARIM/Num_mix30Diversify0/' + 'save_data_changing_nodes.json'
with open(Name,'r') as Data_:
   Data =  json.loads(json.load(Data_))
   
   
   
print(Data)

Path = 'C:/Users/Mahdi/Dropbox/CourseMaterials/PT/'
File = 'Latency_M.png'
E = 'Entropy_M.png'
F_N = Path + File
F_E = Path + E



A = [Data['L_Na'],Data['L_Gr'],Data['L_Im']]

k = 0
for item in A:
    j = 0
    for term in item:
        A[k][j] = term - 0.09
        j = j +1
    k = k +1
        

    
M = 384
L = [0.2,0.4,0.6,0.8,1]

    
PLT = PLOT(L,A,['Nave','Greedy','Low Latency(Imbalance)'],r'$\frac{C}{N}$','Latency(sec)',F_N)


PLT.scatter_line(True)




PLT = PLOT(L,[Data['H_Na'],Data['H_Gr'],Data['H_Im']],['Nave','Greedy','Low Latency(Imbalance)'],r'$\frac{C}{N}$','Entropy(bit)',F_E)


PLT.scatter_line(True)





P1 = 'Diversification_Basic_EXP30'


Data_ = pd.read_csv((P1 + '/'+ 'Analytic.csv'), skipinitialspace = True)
print(Data_)

o = Data_['Anonymity(Im)']

a =  Data_['Anonymity(Gr)']

b =  Data_['Anonymity(N)']

import json

with open(P1 + '/' +'Sim.json','r') as df_sim:
    Data = json.loads(json.load(df_sim))

O = Med(Data['Entropy(Im)'])
A = Med(Data['Entropy(Gr)'])
B = Med(Data['Entropy(Na)'])


Y = [B,b,A,a,O,o]
X = Data_['Tau']

PLT = PLOT(X,Y,['H(m) Naive','H(T) Naive','H(m) Greedy','H(T) Greedy','H(m) Low Latency','H(T) Low Latency'],r'$\tau$','Entropy(bit)',Path + 'Entropy_MT.png')

PLT.scatter_Ent(True)





#################################twin mix delay###################

X = [200,250,300,350,400,450]

t = [0.7,0.7,0.8,0.8,0.9,0.9]
m = [35,54,60,78,85,103]






Y = [t,m]



PLT2 = PLOT(X,Y,[r'Maximum $\tau$','Maximum mixing delay'],'Maximum E2E delay',[r'Randomness ($\tau$)','Mixing delay'],Path + 'Max_D.png')





PLT2.scatter2(True)




###########################################################################################################################

Delay = 50
server = 1
PATH = 'C:/Users/Mahdi/OneDrive/Desktop/server' + str(server) + '/LARIM/Random_Basic_EXP'+  str(Delay) + '/'



import pandas as pd

Data = pd.read_csv(PATH+'Analytic.csv', skipinitialspace= True)

N =128

Name_L = 'Latency_Ran' + str(N) + '.png'

Name_E = 'Entropy_Ran' + str(N) + '.png'


A = [Data['Latancy(N)']-0.003*Delay, Data['Latancy(Gr)']-0.003*Delay,   Data['Latancy(Im)']-0.003*Delay]

B = [Data['Anonymity(N)'], Data['Anonymity(Gr)'],   Data['Anonymity(Im)']]





PLT1 = PLOT(Data['Tau'],A,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r' Randomness ($\tau$)','Latency (sec)',Path + Name_L)

PLT2 =  PLOT(Data['Tau'],B,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r'Randomness ($\tau$)','Entropy (bit)',Path + Name_E)



PLT2.scatter_line(True)

PLT1.scatter_line(True)









####################################################################################################################################################################









server= 1
Delay = 50

PATH1 = 'D:/C/LARIM/Num_Clusters50tau5'+   '/'

NAME1 = 'save_data_changing_clusters.json'

P10 = PATH1 + NAME1

import json

with open(P10  ,'r') as df:
    Data_ = json.loads(json.load(df))


print(Data_['H_Im']+Data_['H_Gr']+Data_['H_Na'])


Y = [Data_['L_Na'],Data_['L_Gr'],Data_['L_Im']]

i = 0
for terms in Y:
    j = 0
    for item in terms:
        Y[i][j] = item - 0.003*Delay
        j = j +1
    i = i +1

print(Y)

X = [2,3,5,10,20,40]


BY = [Data_['H_Na'],Data_['H_Gr'],Data_['H_Im']]

print(Y,BY)

N = 128

Name_L = 'Latency_Cl' + str(N) +'.png'
Name_H = 'Entropy_Cl' + str(N) + '.png'

PLT1 = PLOT(X,Y,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],'K','Latency(sec)',Path + Name_L)

PLT2 =  PLOT(X,BY,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],'K','Entropy(bit)',Path + Name_H)



PLT2.scatter_line(True)

PLT1.scatter_line(True)










Path14 = PATH1 +  'Sim_change_clusters.json'

import json

with open(Path14  ,'r') as df:
    Data14 = json.loads(json.load(df))


print(Med(Data14['Entropy(Na)']))







server= 32
Delay = 20

PATH1 = 'C:/Users/Mahdi/OneDrive/Desktop/server' + str(server) + '/LARIM/Num_mix10Diversify0'+   '/'

NAME1 = 'save_data_changing_nodes.json'

P10 = PATH1 + NAME1

import json

with open(P10  ,'r') as df:
    Data_ = json.loads(json.load(df))


print(Data_)

Y = [Data_['L_Na'],Data_['L_Gr'],Data_['L_Im']]

i = 0
for terms in Y:
    j = 0
    for item in terms:
        Y[i][j] = item - 0.003*Delay
        
        j = j+1
    i = i+1



X = [18,36,54,72,90]


BY = [Data_['H_Na'],Data_['H_Gr'],Data_['H_Im']]

N = 32

Name_L = 'Latency_M' + str(N) + '.png'
Name_H = 'Entropy_M' + str(N) + '.png'

PLT1 = PLOT(X,Y,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],'size','Latency(sec)',Path + Name_L)

PLT2 =  PLOT(X,BY,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],'size','Entropy(bit)',Path + Name_H)



PLT2.scatter_line(True)

PLT1.scatter_line(True)





###########################Simualtion Fig10#####################################################


path = 'C:/Users/Mahdi/OneDrive/Desktop/server1/LARIM/Diversification_Basic_EXP50/'

name = 'Sim.json'





import json

with open(path + name  ,'r') as df:
    Data_Sim = json.loads(json.load(df))
    

Latency = [Data_Sim['Entropy(Im)'],Data_Sim['Entropy(Gr)'],Data_Sim['Entropy(Na)']]
 


Tau = Data_Sim['Tau']



PLT = PLOT(Tau,Latency,['Low Latency','Greedy','Naive'],r'Randomness ($\tau)$','Entropy (bit)',Path + 'Sim_Entropy_Dis128.png')
    



PLT.Box_Plot(True)



#################################################Entropy########################################################################################################################################################################################################################


Name2 = 'Analytic.csv'

import pandas as pd

Data_Prior = pd.read_csv(path + Name2, skipinitialspace = True)


H1 = Data_Prior['Anonymity(Gr)']


H2 = Med(Data_Sim['Entropy(Gr)'])




PLT_E = PLOT(Tau,[H2,H1],[r'$\mathsf{H}(m)$',r'$\mathsf{H}(T)$'],r'$Randomness (\tau)$','Entropy (bit)',Path + 'EntropyTM.png')

PLT_E.colors = ['mediumblue','crimson']
PLT_E.Line_style = ['solid','-.']
PLT_E.scatter_line(True)



###########################FCP vs tau#####################################################

File_name = 'D:/CLAPS/ADV_N/LARIM/Diversification_Advanced_EXP20Fraction_of_corrupted_Path_number_of_Cnodes'


Path10 = File_name + '/' + 'save_data_changing_corrupted_tau.pkl'
import pickle
import numpy as np

with open(Path10, 'rb') as f:
            D_dict1 = pickle.load(f)

Tau = D_dict['tau']

NA = D_dict1['Gr']
Na_BC = NA[0,:].tolist()[0]

print(Na_BC)

Na_CA = NA[1,:].tolist()[0]

Na_FC = NA[2,:].tolist()[0]

Na_R =  NA[3,:].tolist()[0]

Na_WC = NA[4,:].tolist()[0]

NA = [Na_BC, Na_CA, Na_FC, Na_R,Na_WC]


Na_plt = PLOT(Tau,NA,['Worst Case ', 'C closets mix nodes', 'Cluster aware','Random','Best case'],r'Randomness ($\tau$)','FCP',Path + 'FCP_Tau_Dis_Na128.png')
Na_plt.colors = ['navy','red','green','black','orange','cyan']
Na_plt.scatter_line(True)






###########################FCP vs tau#####################################################





Path11 =   File_name + '/' +'1Sim_Corrupted_tau.json'
Path12 =   File_name + '/' +'2Sim_Corrupted_tau.json'
Path13 =   File_name + '/' +'6Sim_Corrupted_tau.json'


import pickle
import numpy as np

with open(Path10, 'rb') as f:
            D_dict = pickle.load(f)


Tau = D_dict['tau']

Im = D_dict['Im']
Im_BC = Im[0,:].tolist()[0]

Im_CA = Im[1,:].tolist()[0]

Im_FC = Im[2,:].tolist()[0]



IM = [Im_BC, Im_CA, Im_FC]


Im_plt = PLOT(Tau,IM,[ 'C closets mix nodes', 'Cluster aware','Random'],r'Randomness ($\tau$)','FCP',Path + 'FCP_Tau_Dis_Im128.png')

Im_plt.colors = ['navy','red','green','black','orange','cyan']
Im_plt.scatter_line(True)

#######
g1 = []
g2 = []
g3 = []
x = 25
GR = []
for i in range(1):
    i = 1
    
    File_name = 'E:/Fig'+str(x)


    Path10 = File_name + '/' + 'save_data_changing_corrupted_tau.pkl'
    import pickle
    import numpy as np

    with open(Path10, 'rb') as f:
                D_dict = pickle.load(f)

    Tau = D_dict['tau']

    Gr = D_dict['Gr']
    GR.append(Gr[0,:].tolist()[0])

    GR.append(Gr[1,:].tolist()[0])

    GR.append(Gr[2,:].tolist()[0])
'''
import numpy as np
gg1 = np.transpose(np.matrix(g1)).tolist()
print(gg1)
gg2 = np.transpose(np.matrix(g2)).tolist()
gg3 = np.transpose(np.matrix(g3)).tolist()

GR = [Med(gg1),Med(gg2),Med(gg3)]
'''
Tau = [0,0.2,0.4,0.6,1]
a = [0.1421,0.1309,0.1043,0.0678,0.0085]
b = [0.012,0.0115,0.0102,0.0094,0.0085]
c = [0.0047,0.0059,0.006,0.007,0.0085]
d = [0.0065,0.006,0.007,0.0071,0.0085]
GR = [a,b,c,d]
Gr_plt = PLOT(Tau,GR,[ 'Worst Case','Single location', 'Diverse location','Random'],r'Randomness ($\tau$)','FCP',Path + 'FCP_Tau_Dis_Gr128.png')

Gr_plt.colors = ['navy','red','green','black','orange','cyan']
Gr_plt.scatter_line(True)

#######

Tau = D_dict['tau']

NA = D_dict['Na']
Na_BC = NA[0,:].tolist()[0]

Na_CA = NA[1,:].tolist()[0]

Na_FC = NA[2,:].tolist()[0]

Na_R =  NA[3,:].tolist()[0]

Na_WC = NA[4,:].tolist()[0]

NA = [Na_BC, Na_CA, Na_FC, Na_R,Na_WC]


Na_plt = PLOT(Tau,NA,['Worst Case ', 'C closets mix nodes', 'Cluster aware','Random','Best case'],r'Randomness ($\tau$)','FCP',Path + 'FCP_Tau_Dis_Na128.png')
Na_plt.colors = ['navy','red','green','black','orange','cyan']
Na_plt.scatter_line(True)



###########################Simulations######################################################################

PTH = 'D:/L/Diversification_Advanced_EXP10FCP'+ '/' +'2Sim_Corrupted_tau.json'
import json

with open(PTH ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Im)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Na)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)',Path + 'sim1_FCP_tau_Dis.png')
    

PLT.Box_Plot(True)


###########################Simulations######################################################################


import json

with open(Path12 ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Im)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Na)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)',Path +'sim2_FCP_tau_Dis.png')
    

PLT.Box_Plot(True)



###########################Simulations######################################################################


import json

with open(Path13 ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Im)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Na)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)',Path +'sim6_FCP_tau_Dis.png')
    

PLT.Box_Plot(True)




###Fcp vs budgetsgaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ###










File_name = 'D:/ADVN/LARIM/Diversification_Advanced_EXP20Fraction_of_corrupted_Path_number_of_Cnodes'


Path10 = File_name + '/' + 'save_data_changing_corrupted_CNodes.pkl'



Path11 =   File_name + '/' +'1Sim_Corrupted_CNodes.json'
Path12 =   File_name + '/' +'2Sim_Corrupted_CNodes.json'
Path13 =   File_name + '/' +'6Sim_Corrupted_CNodes.json'


import pickle
import numpy as np

with open(Path10, 'rb') as f:
            D_dict = pickle.load(f)


Tau = [0.05,0.1,0.15,0.2]


Im = D_dict['Im']
Im_BC = Im[0,:].tolist()[0]

Im_CA = Im[1,:].tolist()[0]

Im_FC = Im[2,:].tolist()[0]

Im_R = Im[3,:].tolist()[0]

Im_WC = Im[4,:].tolist()[0]

IM = [Im_BC, Im_CA, Im_FC, Im_R]
print(IM)


Im_plt = PLOT(Tau,IM,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP',Path + 'FCP_CN_Dis_Im128.png')

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


Gr_plt = PLOT(Tau,GR,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP',Path + 'FCP_CN_Dis_Gr128.png')

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


Na_plt = PLOT(Tau,NA,['Worst Case ', 'Single location', 'Diverse location','Random'],r'Corruption ($\frac{C}{N}$)','FCP',Path + 'FCP_CN_Dis_Na128.png')

Na_plt.colors = ['navy','red','green','black','orange','cyan']
Na_plt.scatter_line(True)



###########################Simulations######################################################################


import json

with open(Path11 ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Na)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Im)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)','sim1_FCP_CN_Dis.png')
    

PLT.Box_Plot(True)


###########################Simulations######################################################################


import json

with open(Path12 ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Na)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Im)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)','sim2_FCP_CN_Dis.png')
    

PLT.Box_Plot(True)



###########################Simulations######################################################################


import json

with open(Path13 ,'r') as df:
    Data_Sim1 = json.loads(json.load(df))
    
    
    
    

Entropy = [Data_Sim1['Entropy(Na)'],Data_Sim1['Entropy(Gr)'],Data_Sim1['Entropy(Im)']]
 


Tau = Data_Sim1['Tau']



PLT = PLOT(Tau,Entropy,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)','sim6_FCP_CN_Dis.png')
    

PLT.Box_Plot(True)

##############################################################################################
####################################Traffic analysis##################################################
##############################################################################################

Path16 = 'D:/T1000/LARIM/Diversification_Basic_EXP50/'
Path17 = 'D:/T5000/LARIM/Diversification_Basic_EXP50/'

Path18 = 'D:/T20000/LARIM/Diversification_Basic_EXP50/'

Name_json = 'Sim.json'


import json

with open(Path16 + Name_json ,'r') as df:
    Data1 = json.loads(json.load(df))
    
with open(Path17 + Name_json ,'r') as df:
    Data2 = json.loads(json.load(df))
    
with open(Path18 + Name_json ,'r') as df:
    Data3 = json.loads(json.load(df))
    
T = Data1['Tau']    


    
Y =[Data1['Entropy(Gr)'],Data2['Entropy(Gr)'],Data3['Entropy(Gr)']]    
    
    
PLTT = PLOT(T,Y,['Traffic rate = 1000','Traffic rate = 5000','Traffic rate = 20000'],r'Randomness ($\tau$)','Entropy (bit)',Path + 'Traffic.png')    
    
    
PLTT.Box_Plot(True)   
    
    



##############################################################################################
##############################################################################################
####################################Traffic analysis##################################################
##############################################################################################

Path19 = 'C:/Users/Mahdi/OneDrive/Desktop/server1/LARIM/Worst_Case_Basic_EXP50'
Name_json = 'Sim.json'


import json

with open(Path18 + Name_json ,'r') as df:
    Data1 = json.loads(json.load(df))
    
    
T = Data1['Tau']    


    
Y =[Data1['Entropy(Im)'],Data1['Entropy(Gr)'],Data1['Entropy(Na)']] 
YY =   [Data1['Latency(Im)'],Data1['Latency(Gr)'],Data1['Latency(Na)']] 
    
    
PLTT = PLOT(T,Y,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Entropy (bit)',Path + 'Sim_Entropy_WC128.png')    
    
    
PLTT.Box_Plot(True)   
    
    

PLTT2 = PLOT(T,YY,['Low Latency','Greedy','Naive'],r'Randomness ($\tau$)','Latency (sec)',Path + 'Sim_Latency_WC128.png')    
    
    
PLTT2.Box_Plot(True)   
    

##############################################################################################

x = [[1,2,3],[4,5,6]]

y = [[3,4,5],[6,7,8]]

T = [0,1]


plt1 = PLOT(x,y,['a','b'],'Latency','Entropy','test_Scatter.png')

plt1.SubPlot_Scatter(T)


plt1.scatt













###########################Size of network######################################################################


Path20 = 'D:/NM1/LARIM/Num_mix50Diversify0/'

Path21 = 'D:/NM5/LARIM/Num_mix50Diversify0/'

Path22 = 'D:/NM9/LARIM/Num_mix50Diversify0/'


name20 = 'save_data_changing_nodes.json'



import json

with open(Path20 + name20,'r') as df1:
    Data20 = json.loads(json.load(df1))
    




with open(Path21 + name20,'r') as df2:
    Data21 = json.loads(json.load(df2))
    







with open(Path22 + name20,'r') as df3:
    Data22 = json.loads(json.load(df3))
    
Delay1 = 50

L = [Data22['L_Gr'],Data21['L_Gr'],Data20['L_Gr']]
for i in range(3):
    j = 0
    for item in L[i]:
        L[i][j] = L[i][j] -3*Delay1/1000
        j = j + 1

H = [Data22['H_Gr'],Data21['H_Gr'],Data20['H_Gr']]

print(L)

X = []

for i in range(5):
    X.append(round(510*(0.2+0.2*i)))

print(X)




PLTN = PLOT(X,[H,L],[r'Entropy ($\tau = 0.9$)',r'Entropy ($\tau = 0.6$)',r'Entropy ($\tau = 0.1$)',r'Latency ($\tau = 0.9$)',r'Latency ($\tau = 0.6$)',r'Latency ($\tau = 0.1$)'],'Mixnet size (N)',[r'Entropy $\mathsf{H}(T)}$ (bit)',r'Latency $\bar{l}_{mix}$ (sec)'],Path + 'Network.png')

PLTN.colors = ['blue','red','green','blue','red','green']
PLTN.markers = ['H','D','v','H','D','v']
PLTN.Line_style = ['-','-','-',':',':',':'] 

PLTN.scatter3(True)






###################################################################################################


T = [0,0.2,0.4,0.45,0.5,0.55,0.6,.65,.7,.75,0.8,1] 



N = 65*3


File_name =  'CLAPS' + str(N)

 

import json

with open('D:/CLAPS/CLAPS128_2/LARIM/CLAPS384' '/' + 'Data_CLAPS.json','r') as dicts_file:
   Data =  json.load((dicts_file))



    
Y = [[3.5923101549548084, 3.712397257819192, 4.18321956679088, 4.306184739228694, 4.435912415692984, 4.5720634035951875, 4.708178132486466, 4.823988231752963, 4.908512808261188, 4.959281452749386, 4.984395078933672, 4.999999999999999],[2.9509153667304417, 3.2192734975073916, 4.02455435688379, 4.220782270896851, 4.4008936071954174, 4.5355393779244135, 4.684887669095659, 4.813128073870923, 4.904426816363271, 4.957274760597682, 4.983397273546687, 4.999999999999999],  [0.5198409276878844, 1.9531766735770375, 3.7675626192663234, 4.264125715989836, 4.526307468500013, 4.723594749596302, 4.850169769494915, 4.91993631026709, 4.960580161673677, 4.982251197527023, 4.993323999357603, 5.0]] 
YY = [[0.06874218392284807, 0.0683714968150339, 0.06579803175367202, 0.06541515219885814, 0.06567752951820993, 0.06709828359606353, 0.07029414721989821, 0.07561537517038079, 0.08297787253385988, 0.09132273940949022, 0.09968959428559601, 0.12953564453124944],[0.05283922502527698, 0.0500372505791313, 0.05214880741555501, 0.05336409407594452, 0.055799188901553776, 0.05866832391876556, 0.063734963439666, 0.07044129405938543, 0.07952403834448699, 0.08883042171216066, 0.0978790921186202, 0.12953564453124944],  [0.041372608957236906, 0.045859450783549194, 0.053822269457409345, 0.05735925710557382, 0.061545059409586614, 0.06663691464795846, 0.07278108971532049, 0.07965206790378326, 0.086823355665852, 0.0942369329901406, 0.10188423525973331, 0.12953564453124947]] 
    
    
PLTT = PLOT(T,Y,['Naive','Greedy','Claps NP'],r'Randomness ($\tau$)','Entropy (bit)',Path + 'ECLAPS.png')    
    
PLTT.scatter_line(True)


    
PLTT1 = PLOT(T,YY,['Naive','Greedy','Claps NP'],r'Randomness ($\tau$)','Latency (sec)',Path + 'LCLAPS.png')    
    
PLTT1.scatter_line(True)
print(Data['Entropy'])





print(Data)











































































###########################################################################################################################

Delay = 50
server = 1
PATH = 'D:/Random_Basic_EXP50Continent2' + '/'



import pandas as pd

Data = pd.read_csv(PATH+'Analytic.csv', skipinitialspace= True)

N =128

Name_L = 'Latency' + '363' + '.png'

Name_E = 'Entropy' + '363' + '.png'


A = [Data['Latancy(N)']-0.003*Delay, Data['Latancy(Gr)']-0.003*Delay,   Data['Latancy(Im)']-0.003*Delay]

B = [Data['Anonymity(N)'], Data['Anonymity(Gr)'],   Data['Anonymity(Im)']]





PLT1 = PLOT(Data['Tau'],A,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r' Randomness ($\tau$)','Latency (sec)',Path + Name_L)

PLT2 =  PLOT(Data['Tau'],B,['Naive Balance','Greedy Balance','Low Latency(Imbalance)'],r'Randomness ($\tau$)','Entropy (bit)',Path + Name_E)



PLT2.scatter_line(True)

PLT1.scatter_line(True)



print(Data)

###########################################################################################################################
###################################################CLAPS VS LARMix#########################################################
###########################################################################################################################

Entropy = [[4.050806067055514, 4.042877750863812, 5.1382255087412645, 5.395100302652372, 5.700051431476169, 5.994918584534938, 6.308988274400701, 6.609711745742906, 6.8389251482628675, 6.953904193912425, 6.990221354728269, 7.000000000000001], [5.538416408853504, 5.633798344816049, 5.919322128106181, 6.0136519208182015, 6.133450118920855, 6.284758791949617, 6.470202974533496, 6.679397640357855, 6.862636954758155, 6.96010513287689, 6.991676769540394, 7.000000000000001], [0.7170781278299563, 3.2262691181502285, 5.339774881117563, 5.886201350347083, 6.309936510804676, 6.598361098677435, 6.781294685191199, 6.884167302897218, 6.944184627245784, 6.974601447534486, 6.9898896052715465, 7.0]]
Latency = [[0.039062077464281246, 0.033267706720531275, 0.03445217446291051, 0.034873819801437195, 0.036326476231914574, 0.03850788143951403, 0.04249518618171673, 0.04964346118641945, 0.06265446732581252, 0.0779241185818915, 0.09185555894779786, 0.12076518249512405], [0.06349518479339891, 0.06291984854742018, 0.05951899646314892, 0.05848195119575045, 0.05743714792089552, 0.05669990180284677, 0.0567783861811009, 0.060150085060848284, 0.06950038751249574, 0.08229508738751345, 0.09459465183500962, 0.12076518249512405], [0.02630505011373131, 0.02982162330322775, 0.03941969130803672, 0.042873607213794046, 0.04676580984994805, 0.05128001038013476, 0.056890013853168234, 0.06371393111169513, 0.07148247207073749, 0.08000466235350892, 0.08849028272315337, 0.12076518249512404]]
Time = [[4.644550800323486, 4.96944797039032, 4.927890062332153, 4.914297461509705, 4.897388219833374, 4.881376504898071, 4.870422959327698, 4.8767359256744385, 4.83575701713562, 4.81324303150177, 4.820991277694702, 4.199455618858337], [4.202556133270264, 4.211218953132629, 4.190826058387756, 4.200228691101074, 4.184936285018921, 4.189448952674866, 4.197073936462402, 4.205764889717102, 4.183181524276733, 4.16966712474823, 4.189114928245544, 4.22226083278656], [1290.754358291626, 1293.0408383607864, 1289.6205178499222, 1288.7804468870163, 1309.2851876020432, 1297.7925153970718, 1303.5202704668045, 1293.5721267461777, 1307.7570661306381, 1300.9157946109772, 1291.776258111, 1288.2180285453796]]

E = [Entropy[1],Entropy[0],Entropy[2]]
L = [Latency[1],Latency[0],Latency[2]]
Tau = [0,0.2,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,1]
E_ = [[],[],[]]
L_ = [[],[],[]]
Tau_ = []
A = [3,4,5,7,8,9]
for i in range(len(Tau)):
    if not (i in A):
        Tau_.append(Tau[i])
        E_[0].append(E[0][i])
        L_[0].append(L[0][i])
        E_[1].append(E[1][i])
        L_[1].append(L[1][i])        
        E_[2].append(E[2][i])
        L_[2].append(L[2][i])    
    
print(E_,L_,Tau_)
W = 128

Name_L = 'Latency' + 'ClapsMix'+'LARMix'+''+str(W) + '.png'

Name_E = 'Entropy' + 'ClapsMix'+'LARMix'+'' +str(W)+ '.png'

Name_L_E = 'E_L' + 'ClapsNix'+'LARMix'+'' +str(W)+ '.png'



PLT1 = PLOT(Tau,E,['Naive ALGO','Greedy ALGO','CLAPSMix'],r' Randomness ($\tau$)','Entropy (bit)',Path + Name_E)


PLT2 = PLOT(Tau,L,['Naive ALGO','Greedy ALGO','CLAPSMix'],r' Randomness ($\tau$)','Latency (sec)',Path + Name_L)

PLT3 = PLOT(L,E,['Naive ALGO','Greedy ALGO','CLAPSMix'],'Latency (sec)','Entropy (bit)',Path + Name_L_E)

PLT2.scatter_line(True)

PLT1.scatter_line(True)

PLT3.scatter_line2(True)

time = [[],[],[]]

Time= [[0.10197699069976807, 0.10996878147125244, 0.09966635704040527, 0.10978817939758301, 0.10979008674621582, 0.10197293758392334, 0.10200512409210205, 0.10978889465332031, 0.10980403423309326, 0.10979688167572021, 0.11002767086029053, 0.08690035343170166], [0.08634138107299805, 0.09414291381835938, 0.09946656227111816, 0.09415435791015625, 0.09416580200195312, 0.10197460651397705, 0.10179579257965088, 0.10197556018829346, 0.09415805339813232, 0.09416711330413818, 0.09415996074676514, 0.09415650367736816], [3.8956069946289062, 3.773792028427124, 3.788954973220825, 3.792136311531067, 3.8314276933670044, 3.805463433265686, 3.8283724784851074, 3.8062692880630493, 3.8215184211730957, 3.8052529096603394, 3.835369348526001, 3.8349238634109497]]

for i in range(len(Time[1])):
    time[1].append(1)
    
    time[0].append(int(10*Time[0][i]/Time[1][i])/10)
    time[2].append(int(10*Time[2][i]/Time[1][i])/10)
print(time)


Frac1 = [0]*len(E_[0])
Frac2 =[0]*len(E_[0])

Frac3 = [0]*len(E_[0])
for i in range(len([3])):
    for j in range(len(E_[0])):
        Frac1[j]= E_[i+0][j]/L_[i+0][j]
        
Frac = [Frac2,Frac3,Frac1]        
print(len(Frac1),len(E[0]))        
Name_Frac = 'Frac' + 'ClapsMix'+'LARMix' + '_' +str(W)+ '.png'

PLT4 = PLOT(Tau_,Frac,['Greedy','CLAPS','Naive'],r' Randomness ($\tau$)','Entropy/Latency',Path + Name_Frac)
print(Frac)
PLT4.colors[0] = 'red'
PLT4.colors[1] = 'green'
PLT4.colors[2] = 'royalblue'
PLT4.scatter_line(True)























