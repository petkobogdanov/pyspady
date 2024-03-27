<div align="center">
  <img src="assets/pyspady-logo.png" alt="PySpady Logo">
</div>

# PySpady: A Python sparse multi-dictionary coding library
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CACEA?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![TensorLy](https://img.shields.io/badge/TensorLy-FF6F61?style=for-the-badge&logo=tensorly&logoColor=white)](https://tensorly.org/stable/index.html)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SKLearn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MatPlotLib](https://img.shields.io/badge/Matplotlib-239120?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
## Authors
- [Michael J. Paglia](https://github.com/michaelpaglia)
- [Proshanto Dabnath](https://github.com/proshantod)
- [Michael A. Smith](https://github.com/Homercat1234)
- [Joseph Regan](https://github.com/reganjoseph)

## Introduction

PySpady enables users from all disciplines to leverage state-of-the-art and classical sparse encoding algorithms and methodologies to jointly model **spatial-temporal data** by graph and temporal dictionaries. The current implementation efficiently exploits both structural graph regularities and temporal patterns encoded within **2D temporal graph signals** (McNeil et. al, 2021) and more generally any **multi-way tensors** (McNeil and Bogdanov, 2023) with priors on all or a subset of modes.

## Current features
- [x] Missing value imputation
- [x] Future value prediction
- [x] Automatic hyperparameter optimization
- [x] [![JSON](https://img.shields.io/badge/JSON-8A2BE2?style=flat&logo=json&logoColor=white)](https://www.json.org/json-en.html) [![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/) user input configurations
- [x] Outlier detection with visualizations
- [x] Community detection with visualizations
- [x] Command-line interface
- [x] Demonstration datasets    

## Installation
[Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository's main branch and run the following terminal command in the PySpady directory to download the required dependencies:
```
pip3 install .
```
Please make sure you already have the latest version of [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) and [![Pip](https://img.shields.io/badge/pip-3776AB?style=flat&logo=pypi&logoColor=white)](https://pypi.org/) installed.

### Datasets

There are numerous datasets available for use in the PySpady library. Due to storage limitations, certain datasets cannot be stored on the Git repository and must be downloaded externally.

- [New York City taxi pickup and dropoff data, circa 2017: 2D temporal graph signal]()
- [New York City taxi pickup and dropoff data, circa 2017: Multi-way tensors]()
- [Multi-way tensors synthetic data]()

2D temporal graph signal synthetic data is already available.


--------
# pyspady
PySpady - a Python sparse multi dictionary coding library

Done
optimization 
* TGSD
  * vannila
* MDTD
  * vannila 
  

Implementation
* dictionaries
  


Testing

Todo

1. Some dictionaries
   
3. Front end
   1. data formats (mask as well)
    * matrix input
    * row,col, etc. value inputs 
    * enter csv
    * pandas dataframe
   1. make defaul parameters
   2. config parameters 
   3. how to return missing values 
  
4. Auto config 
   1. auto parameter selections 
   2. graph learning
   3. estimate time to complete 
   4. return explanatory figs?
      
5. Different optimizers 
   1. gradient based
   2. OMP for differnt mod
      
6. Other Models  
   1. 2D-OMP
   2. Low Rank Dictionary Selection 
   3. Different regualriers
   4. coloums sparsity
   5. Can we implement non-negataive on  a dictionary times co-efficents?
      
7. Dictionary selection
   1. 1D Screening
   2. Scaling to large graphs
      
8. Prep datasets as use cases 
  
-------------------------------------------------------------------------------------

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
    

    

    
    
  
