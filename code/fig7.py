import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import qmc
from scipy.stats import distributions
import statistics
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import random
import sympy as sp
import os
import argparse
from scipy.signal import argrelextrema
from sympy.utilities import lambdify
import time
import multiprocessing

df = []

def gen_matrix_F(n,vr):
    x, y = n, vr
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)
    A = -I + B
    D = np.zeros((n, n))
    return A, D

# code for sparsity > 0, dictionary = {network size:n of 0 entries}
# def gen_matrix_F(n, vr):
    #sparsity_dict = {3: , 4: , 5: , 6: , 7: , 8: , 9: , 10: }
    #I = np.eye(n)
    #mu = 0
    #B = mu + np.sqrt(vr) * np.random.randn(n, n)
    #np.fill_diagonal(B, 0)
    #A = -I + B
    #mask = np.eye(n) == 0
    #od = np.sum(mask)
    #s = sparsity_dict[n]
    #odi = np.where(mask)
    #indices_to_zero = np.random.choice(od, s, replace=False)
    #for index in indices_to_zero:
        #i, j = odi[0][index], odi[1][index]
        #A[i, j] = 0
    #D = np.zeros((n, n))
    #return A, D

def process_tuple(tuplet):
    n,vr = tuplet
    rm = []
    for i in range(1000):
        m = gen_matrix_F(n,vr)[0]
        rm.append(m)

    srm = []
    for m in rm:
        ev = np.linalg.eigvals(m)
        if np.max(np.real(ev)) < 0:
            srm.append(m)
    
    D = gen_matrix_F(n,vr)[1]
    D[0, 0] = 1
    D[1, 1] = 100
    k = np.arange(0, 101, 0.2)
  
    urm = []
    t1a = []
    t1ar = []
    t1ai = []
    #
    t1b = []
    t1br = []
    t1bi = []
    #
    t2a = []
    t2ar = []
    t2ai = []
    #
    t2b = []
    t2br = []
    t2bi = []

    for m in srm:
        Em = []
        Emi = []
        for i in range(len(k)):
            R = m - D * (k[i] ** 2)
            eigval = np.linalg.eigvals(R)
            Em.append(np.max(np.real(eigval)))
            idx_max = np.argmax(np.real(eigval))
            Emi.append(np.imag(eigval[idx_max]))
        a = np.max(Em)
        index = np.argmax(Em)
        nEm = np.array(Em)
        if a > 0:
            if Emi[index] == 0:
                numZeroCrossing = np.count_nonzero(np.diff(np.sign(Em)))  # Count zero crossings
                numpositivelocalmaxima = np.sum(nEm[argrelextrema(nEm,np.greater)]>0) > 0   
                if numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 0:
                    t1a.append(m)
                    t1ar.append(Em)
                    t1ai.append(Emi)
                elif numpositivelocalmaxima > 0 and numZeroCrossing == 1:
                    t1b.append(m)
                    t1br.append(Em)
                    t1bi.append(Emi)
                elif numpositivelocalmaxima == 0 and numZeroCrossing % 2 == 1:
                    t2a.append(m)
                    t2ar.append(Em)
                    t2ai.append(Emi)
                elif numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 1:
                    t2b.append(m)
                    t2br.append(Em)
                    t2bi.append(Emi)
    percent = (len(t1a)+len(t1b))*0.1
    
    return percent

if __name__ == "__main__":
    # list of tuples
    #tuplet_list = [(n1, var1), (n2, var2), ..., (nN, varN)]  
    n = list(range(3, 30))                                      # example: n up to 30
    var = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]   # example: var 0.1 increment 
    dp_list = [(x, y) for x in n for y in var] 
    start_time = time.time()
    # Number of cores to use
    num_cores = 3

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_cores)

    # Use the pool to apply the process_tuple function to each tuple in the list
    results = pool.map(process_tuple, dp_list)
    # Close the pool 
    pool.close()
    pool.join()
    end_time = time.time()

    #df.extend(results)
    df_data = pd.DataFrame({'n': [x[0] for x in dp_list], 'var': [x[1] for x in dp_list], 'Percentage': results})
    df_data.to_csv(os.path.join('/Users/aibekk99/Desktop/repository/data/', 'z.csv'), index=False)
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
