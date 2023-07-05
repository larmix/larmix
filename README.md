The repository contains code to reproduce results for the paper titled "LARMix: latency aware routing in mixnets". However, one can easily modify the code and parameters to build on top of the existing results.

## Initial setup and dependencies
The code can be executed on any standard laptop or desktop device running Ubuntu 18.04 or higher. It has been tested to work for Python 3.8.10. The dependencies required for executing the code are as follows:

- fpdf==1.7.2
- kaleido==0.2.1
- matplotlib==3.5.2
- numpy==1.21.2
- pandas==1.3.2
- plotly==5.10.0
- pulp==2.7.0
- scikit_learn==1.1.1
- scikit-learn-extra==0.2.0
- scipy==1.8.1
- simpy==4.0.1
- tabulate==0.9.0

These dependencies can be easily installed using the following command:
`pip install -r requirements.txt`

## Code execution
The file `Main.py` contains the code for producing all the results mentioned in LARmix evaluations.

- Just executing `python3 Main.py` will automatically perform all the experiments and generate results in `Figures` and `Tables` folder. 

Specifically, the following results can be generated:

- Figure 4a (Diversification) and Figure 5
  - `C.Basic_Exp_Diversification('Fig4_a', 128*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'>

- Figure 4b (Random Assignment)
  - `C.Basic_Exp_Random('Fig4_b', 128*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'> 

- Figure 4c (Bad Assignment)
  - `C.Basic_Exp_WC('Fig4_c', 128*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'>

- Table 3 (also displayed in the command line)
  - `C.Table3(0.2, 'Table3', 128*3)`
  - Inputs: <'end-to-end delay (in seconds)', 'name of experiment', 'size of mixnet (nodes*layers)'>

- Figure 6 (Maximum Tau or Mu)
  - `C.Maximum_tau_mu([0.25, 0.3, 0.4], 128*3)`
  - Inputs: <'list of end-to-end delays (in seconds)', 'size of mixnet (nodes*layers)'>

- Figure 7 (Network Size variation)
  - `C.Network_Size('Fig7', 300, 2)`
  - Inputs: <'name of experiment', number of mixnodes, 'no. of iterations'>

- Figures 8 (Fraction of corrupted paths vs tau) and 10 (Entropy vs tau)
  - `C.FCP_Greedy('Fig8and10', 32*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'>

- Figure 9 (fraction of corrupted paths vs fraction of corrupted nodes)
  - `C.FCP_Cnodes('Fig9', 32*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'>

- Figure 11 (LARMix vs CLAPS)
  - `C.Claps('Fig11', 32*3, 2)`
  - Inputs: <'name of experiment', 'size of mixnet (nodes*layers)', 'no. of iterations'>
 
- Table 4 (LARMis vs 2-layer vanialla mixnet)
  - `C.Two_Layers_VS_LARMIX('Loopix_Larmix',128,1)`
  -  Inputs: <'name of experiment', 'no. of nodes in each layer', 'no. of iterations'>

## Additional Notes

- After running each experiment, the corresponding figures will be automatically saved in the "Figures" folder. For experiments generating tables, the table will be saved in a PDF file in the "Tables" folder.

- For each experiment, you need to input the experiment name, the size of the mix network, and the number of iterations. In Main.py, we have provided initial values to ensure results similar to the paper, but they can be modified as needed. 

- Ensure that the network size is a factor of 3 and preferably larger than 3*16 to obtain reasonable results. 

- For Figure 7, network size is a factor of 10. 

- Increasing the number of iterations improves accuracy and reduces sampling errors for better results. However, it increases execution time. Thus, we have kept the iterations to two, but they can be increased to a value of 1000 (as in the main paper) at the expense of more than 24 hours to complete the experiment and obtain the results.

- It is recommended to give each experiment a different name to avoid any conflicts when running the codes in parallel. Upon running each experiment, a file will be generated with its corresponding name to store necessary data used for plotting.

- There are other additional parameters, such as the clustering method, the rate of traffic etc., that can be modified in the file `LARMIX.py`. We have the default values to preserve the parameters used in the original paper. 

