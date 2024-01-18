# pyspady
PySpady - a Python sparse multi dictionary coding library

Done
optimization 
* TGSD
  * vannila
2 MDTD
  a vannila 
  

Implementation
1 dictionaries
  a 
2 

Testing

Todo

1 Some dictionaries
2 Front end
  a data formats (mask as well)
    i matrix input
    ii row,col, etc. value inputs 
    iii enter csv 
  b make defaul parameters
  c config parameters 
  d how to return missing values 
  
3 Auto config 
  a auto parameter selections 
  b graph learning
  c estimate time to complete 
  d return explanatory figs? 
4 Different optimizers 
  a gradient based
  b OMP for differnt mods
5 Other Models  
  a 2D-OMP
  b Low Rank Dictionary Selection 
  c Different regualriers
    i coloums sparsity 
6 Dictionary selection
  a 1D Screening  
7 Prep datasets as use cases 
  


TODO
* Boya and Max, can you try to design/describe the interfaces for key functions/classes. For example, we will have something for dictionry generation, then we will have diffeernt solvers for TGSD: 2D OMP and L1, tensors, etc
* Different models: low rank: LYWR (as in TGSD)  or via a rank norm LWR with rank-norm(W); versus non-low-rank as in the separable dictionaries paper or 2D-OMP
* Different optimizers (coding): ADMM (as in TGSD), FASTI or gradient based (from the separable dictionary paper), OMP greedy, Kronecker + vetorize for 1D problem.  



Dictionary Generate Methods atoms should correspond to coloumns 

gen_GFT
  Params:
    numpy array: adjacency matrix
    bool: Normalize 
  Returns:
    a graph fourier transform matrix which is normalized if noramlize true

gen_DFT
  Params:
    int: number of timesteps 
  Returns:
    discrete fourier transform matrix 

gen_Rama
  Params:
     int: number of timesteps 
     int: Max period
   Returns:
     Ramanujan periodic dicitonary 


TGSD
  Params: 
    Left dictionary (Psi)
    Bool describe Orthogonality of Psi
    Right dicitonary (Phi)
    Bool describe Orthogonality of Phi
    Data 
    Missing Value Mask 
    Termination condition
      Fit tolerance
      Max iterations 
    Lambda values 
    Rao values
  Returns:
    Y,Z
    W,U
    completed matrix if appopriate 
    

    

    
    
  
