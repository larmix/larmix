# -*- coding: utf-8 -*-
"""
    plot
"""
def compatible(A,B):

    a = len(A)
    b = len(B)
    X = []
    
    Min = min(a,b)
    if not a > Min:
        i = 0
        for item in A:
            
            X.append(B[i])
            i =i +1
        return A,X
    else:
        i = 0
        for item in B:
            X.append(A[i])
            i = i+1
        return X,B  

def CDF_Probability(data,T):
    import numpy as np
    data = np.array(data)
    return (1-np.sum(data >= T) / data.size)           
                                

class PLOT(object):
    
    def __init__(self,X,Y,Descriptions,X_label,Y_label,name,condition = False):        
        self.X = X
        self.Y = Y
        self.Desc = Descriptions
        self.XL =X_label
        self.YL = Y_label
        self.name = name
        self.condition = condition  
        self.markers = ['H','D','v','^','<','>','d']
        self.Line_style = ['-',':','--','-.','steps']           
        self.colors = ['royalblue','red','green','fuchsia','cyan','indigo','teal','lime','blue','black','orange','violet','lightblue']
    def rectangle(self,i):
     
        import numpy as np
                 
        import matplotlib.pyplot as plt 
        plt.close('all')  
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D        
        legend_elements = []

        legend_elements.append(Line2D([0], [4], color=self.colors[i], lw=4, label=self.Desc[0]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements)         
        
        plt.plot(self.X,self.Y[0],  alpha=1,color=self.colors[i],linestyle=self.Line_style[i], lw=8)
        plt.scatter(self.X,self.Y[0], marker = '*',linewidths =0.16, alpha=1,color=self.colors[i], lw=8)
        font1 = {'family':'Times Roman','color':'b','size':20}

        plt.xlabel(self.XL,fontdict = font1)
        plt.ylabel(self.YL,fontdict = font1)

        plt.savefig(self.name,format='png', dpi=600)     
    
    
    
    
    def scatter_line(self,Grid,Log = False):
        import numpy as np
                 
        import matplotlib.pyplot as plt    

                 
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [4], color=self.colors[i], lw=4, label=self.Desc[i],linestyle=self.Line_style[i]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements,fontsize = 12, loc = 'upper right') 
        if Log:
            plt.xscale("log")           
        if Grid:
            plt.grid(linestyle='--')   
   
        for j in range(len(self.Y)):            
            plt.plot(self.X,self.Y[j],  alpha=1,color=self.colors[j],linestyle=self.Line_style[j])
            plt.scatter(self.X,self.Y[j], marker = 'h',linewidths =0.16, alpha=1,color=self.colors[j])

        plt.ylim(0,1.25*np.max(self.Y[0]))
        plt.xlim(0, 1.*np.max(self.X)) 

        
        font1 = {'family':'Times Roman','color':'black','size':20}

        plt.xlabel(self.XL,fontdict = font1,fontsize = 17, fontweight = 'bold')
        plt.ylabel(self.YL,fontdict = font1,fontsize = 17, fontweight = 'bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.xticks(fontsize=15, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.name,format='png', dpi=600) #save the plpoted dots  





    def scatter_line2(self,Grid,Log = False):
        import numpy as np
                 
        import matplotlib.pyplot as plt    

                 
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [4], color=self.colors[i], lw=4, label=self.Desc[i],linestyle=self.Line_style[i]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements,fontsize = 12, loc = 'lower right') 
        if Log:
            plt.xscale("log")           
        if Grid:
            plt.grid(linestyle='--')   
   
        for j in range(len(self.Y)):            
            plt.plot(self.X[j],self.Y[j],  alpha=1,color=self.colors[j],linestyle=self.Line_style[j])
            plt.scatter(self.X[j],self.Y[j], marker = 'h',linewidths =0.16, alpha=1,color=self.colors[j])

        plt.ylim(0,1.05*np.max(self.Y[j]))
        plt.xlim(0.02, 1.*np.max(self.X[j])) 

        
        font1 = {'family':'Times Roman','color':'black','size':20}

        plt.xlabel(self.XL,fontdict = font1,fontsize = 17, fontweight = 'bold')
        plt.ylabel(self.YL,fontdict = font1,fontsize = 17, fontweight = 'bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.xticks(fontsize=15, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.name,format='png', dpi=600) #save the plpoted dots 










    def scatter_Ent(self,Grid,Log = False):
        import numpy as np
                 
        import matplotlib.pyplot as plt  
        self.colors = ['navy','royalblue','darkred','red','green','lime']

                 
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [4], color=self.colors[i], lw=4, label=self.Desc[i]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements,fontsize = 12) 
        if Log:
            plt.xscale("log")           
        if Grid:
            plt.grid(linestyle='--')   
   
        for j in range(len(self.Y)):            
            plt.scatter(self.X,self.Y[j], marker = 'h',linewidths =0.16, alpha=1,color=self.colors[j])

        plt.ylim(0,1.1*np.max(np.matrix(self.Y)))
        plt.xlim(0, 1.1*np.max(self.X)) 

        
        font1 = {'family':'Times Roman','color':'black','size':20}

        plt.xlabel(self.XL,fontdict = font1)
        plt.ylabel(self.YL,fontdict = font1)
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.xticks(fontsize=13, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=13)
        plt.tight_layout()
        
        plt.savefig(self.name,format='png', dpi=600) #save the plpoted dots  
    def CDF(self):
        import matplotlib.pyplot as plt                        
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for j in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [4], color=self.colors[j], lw=4, label=self.Desc[j]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements) 
       
        for i in range(len(self.Y)):
            Axis_Y = self.Y[i]
            CDF = []
            INDX = []
            for j in range(1001):
                Index = CDF_Probability(Axis_Y,j/1000)
                CDF.append(Index)
                INDX.append(j/1000)
                
            plt.plot(INDX,CDF,  alpha=1,color=self.colors[i])

 

        
        font1 = {'family':'Times Roman','color':'b','size':10}
        fig, axs = plt.subplots()
        plt.xlabel(self.XL,fontdict = font1)
        plt.ylabel(self.YL,fontdict = font1)                                
        plt.savefig(self.name,format='png', dpi=600)  #save the plpoted dots          

    def Box_Plot(self,Grid): 
        color1 = self.colors[0]
        color2 = self.colors[2]
        self.colors[0] = color2
        self.colors[2] = color1
        import matplotlib.pyplot as plt        
        plt.close("all")             
        from matplotlib.lines import Line2D
        
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['xtick.labelsize']= 15
        plt.rcParams['ytick.labelsize']= 15
        '''plt.xticks(fontsize=15, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=15)'''
        
        font1 = {'family':'Times Roman','color':'black','size':12}
        fig, axs = plt.subplots()
        axs.set_ylabel(self.YL,fontdict = font1, fontsize='x-large', fontweight='bold'
)
        axs.set_xlabel(self.XL,fontdict = font1,fontsize='x-large', fontweight='bold'
                     
)
        flierprops = dict(marker='x', markersize=2)#, markeredgecolor='b')
        medianprops = dict(color="black",linewidth=1.5)
        whiskerprops = dict(linewidth=1.5)
        capprops = {'linewidth': '1.5'}
        Elements = []
        for j in range(len(self.Y)):
            Elements.append(Line2D([0], [4], color=self.colors[j], lw=4, label=self.Desc[j]))
            for i in range(len(self.X)): 
                if j==1:                  
                    axs.boxplot(self.Y[j][i], labels = [self.X[i]],positions= [0.9*i + 0.3*j]
                            , notch=False, patch_artist=True,
                            boxprops=dict(facecolor=self.colors[j], color=self.colors[j]

),
        
                            #capprops=dict(color=self.colors[j+2]),
                            #whiskerprops=dict(color=self.colors[j+1]),
                            #medianprops=dict(color= self.colors[j+2]),
        flierprops = dict(marker='x', markersize=2),#, markeredgecolor='b')
        medianprops = dict(color="black",linewidth=1.5),
        whiskerprops = dict(linewidth=1.5)
                            
                            
                            
        )
                else:
                         axs.boxplot(self.Y[j][i], labels = [''], positions= [0.9*i + 0.3*j]
                            , notch=False, patch_artist=True,
                            boxprops=dict(facecolor=self.colors[j], color=self.colors[j]

),
                                                      #whiskerprops=dict(color=self.colors[j+1]),
                            #medianprops=dict(color= self.colors[j+2]),
        flierprops = dict(marker='x', markersize=2),#, markeredgecolor='b')
        medianprops = dict(color="black",linewidth=1.5),
        whiskerprops = dict(linewidth=1.5)
                            
                            
                            
        )
                
        

        '''fig.set_size_inches(18.5, 10.5, forward=True)'''
        legend_elements = Elements
        axs.legend(handles=legend_elements,fontsize = 12, loc = 'lower right') 
        if Grid:
            plt.grid(linestyle='--')
        #plt.ylim(0,12)   
        plt.xlabel(self.XL,fontdict = font1)
        plt.ylabel(self.YL,fontdict = font1)
        
        plt.tight_layout()            

            
        '''axs.tick_params(axis='x', labelsize=15)
        axs.tick_params(axis='y', labelsize=15)        '''
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()        
        
        
    def scatter2(self,Grid,Log = False):
        
         
         
        import numpy as np
                 
        import matplotlib.pyplot as plt  


                 
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [4], color=self.colors[i], lw=4, label=self.Desc[i]))
                           

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements,fontsize = 11) 
        if Log:
            plt.xscale("log")           
        if Grid:
            plt.grid(linestyle='--')   





  
         
        ax.scatter(self.X,self.Y[0], marker = 'h',linewidths =0.16, alpha=1,color=self.colors[0])
        ax.set_ylabel(self.YL[0], fontsize=17)
        # create the second scatter plot and set the y-axis label
        ax2 = ax.twinx()
        ax2.scatter(self.X,self.Y[1], marker = 'h',linewidths =0.16, alpha=1,color=self.colors[1]) 
        ax2.set_ylabel(self.YL[1], fontsize=17)
        ax.set_xlabel(self.XL, fontsize=17)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)        
        
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)        
        
        
        plt.ylim(0,1.1*np.max(np.matrix(self.Y)))
        plt.xlim(0.9*np.min(self.X), 1.1*np.max(self.X)) 

        
        font1 = {'family':'Times Roman','color':'black','size':20}

        plt.xlabel(self.XL,fontdict = font1)
        
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        '''plt.xticks(fontsize=13, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=13)'''
        plt.rcParams['xtick.labelsize']= 20
        plt.rcParams['ytick.labelsize']= 20
        plt.tight_layout()         
        plt.savefig(self.name, format='png', dpi=600)
         
         
         




    def scatter3(self,Grid,Log = False):
        
         
         
        import numpy as np
                 
        import matplotlib.pyplot as plt  


                 
        plt.close('all')
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements1 = []
        legend_elements2 = []        
        for i in range(3):
            legend_elements1.append(Line2D([0], [4], marker = self.markers[i], color=self.colors[i], lw=4, label=self.Desc[i],linestyle=self.Line_style[i]))
            legend_elements2.append(Line2D([0], [4], marker = self.markers[i+3], color=self.colors[i+3], lw=4, label=self.Desc[i+3],linestyle=self.Line_style[i+3]))                           

        # Create the figure
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.legend(handles=legend_elements1,fontsize = 7.8, loc = 'upper left') 
        ax2.legend(handles=legend_elements2,fontsize = 7.8, loc = 'upper right')        
        
        
        if Log:
            plt.xscale("log")           
        if Grid:
            plt.grid(linestyle='--')   




        for i in range(len(self.Y[0])):
            
  
         
            ax.scatter(self.X,self.Y[0][i], marker = self.markers[i],linewidths =0.25, alpha=1,color=self.colors[i])
            ax.plot(self.X,self.Y[0][i],  alpha=1,color=self.colors[i],linestyle=self.Line_style[i])
        
        # create the second scatter plot and set the y-axis label
        
            ax2.scatter(self.X,self.Y[1][i], marker = self.markers[i+3],linewidths =0.25, alpha=1,color=self.colors[i+3]) 
            ax2.plot(self.X,self.Y[1][i],  alpha=1,color=self.colors[i+3],linestyle=self.Line_style[i+3])        
        
        
        ax.set_ylabel(self.YL[0], fontsize=17)
        ax2.set_ylabel(self.YL[1], fontsize=17)
        ax.set_xlabel(self.XL, fontsize=17)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)        
        
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)        
        

        ax.set_ylim([0,10]) 
        ax2.set_ylim([0,0.15])

       # plt.ylim(0,1.05*np.max(self.Y[j]))
       # plt.xlim(0.02, 1.*np.max(self.X[j])) 
        
        font1 = {'family':'Times Roman','color':'black','size':20}

        plt.xlabel(self.XL,fontdict = font1)
        
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        '''plt.xticks(fontsize=13, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=13)'''
        plt.rcParams['xtick.labelsize']= 20
        plt.rcParams['ytick.labelsize']= 20
        plt.tight_layout()         
        plt.savefig(self.name, format='png', dpi=600)





























         
      
        
    def Scatter_Interpolation(self,Grid,scatter,T):
        
        import matplotlib.pyplot as plt
        plt.close("all")
        from matplotlib.lines import Line2D
        legend_elements = []

        fig, axs = plt.subplots()
        axs.set_ylabel(self.YL)
        axs.set_xlabel(self.XL)         
        fig.set_size_inches(18.5, 10.5, forward=True)
        
        from scipy.interpolate import interp1d
        import numpy as np
        term = 0
        for I in T:
            i = int(I)
            xs,ys = compatible(self.X[i],self.Y[i])
            maxx = max(xs)
            minx = min(xs)
            interp_func = interp1d(xs, ys)            
            if scatter:
                plt.scatter(xs,ys,s = 30, color= self.colors[term], alpha=0.5)            
            interval = maxx-minx
            Int_Y = interp_func(np.arange(minx, maxx,interval/100))
            plt.plot(np.arange(minx, maxx,interval/100),Int_Y,  alpha=1,color=self.colors[term])     
            legend_elements.append(Line2D([0], [4],color=self.colors[term], alpha=0.75, lw=4, label=r'$\tau =  $' + str(i/5)))            
            term = term +1
  
        axs.legend(handles=legend_elements) 
        plt.savefig(self.name, format='png', dpi=600)
        if Grid:
            plt.grid()  
        plt.show()  

    def SubPlot_Scatter(self,T):
        import matplotlib.pyplot as plt
        plt.close("all")
        from matplotlib.lines import Line2D
        legend_elements = []
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.xticks(fontsize=13, weight = 'bold')
        plt.yticks(weight = 'bold', fontsize=13)
        plt.rcParams['xtick.labelsize']= 15
        plt.rcParams['ytick.labelsize']= 15



        fig, axs = plt.subplots(len(T), sharex=True, sharey=True)
        fig.set_size_inches(20,15, forward=True)      
        term = 0
        for i in T:
            xs,ys = compatible(self.X[i],self.Y[i])

            axs[term].scatter(xs,ys,s = 30, color= self.colors[term], alpha=0.9)
           
            axs[term].set_title(r'$\tau =  $' + str(i/5),fontsize=20)           
            
            term = term + 1

            '''axs.set_ylabel(self.YL, fontsize='x-large', fontweight='bold'
)
            axs.set_xlabel(self.XL, fontsize='x-large', fontweight='bold'
                     
)        '''
         
        for ax in axs.flat:
            ax.set_ylabel(self.YL, fontsize='x-large', fontweight='bold'
)
            ax.set_xlabel(self.XL, fontsize='x-large', fontweight='bold'
                     
) 
            '''ax.set(xlabel=self.XL, ylabel=self.YL)
            ax.grid()
            ax.tick_params(axis='x', labelsize=40)
            ax.tick_params(axis='y', labelsize=40)'''
            
        
        plt.tight_layout()
               
        plt.show()
        plt.savefig(self.name,format='png', dpi=600)        
        


