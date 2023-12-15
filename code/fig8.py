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
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)
    A = -I + B
    D = np.zeros((n, n))
    return A, D
def process_tuple(tuplet):
    x,y = tuplet
    rm = []
    for i in range(1000):
        m = gen_matrix_F(3,0.35)[0]
        rm.append(m)

    srm = []
    for m in rm:
        ev = np.linalg.eigvals(m)
        if np.max(np.real(ev)) < 0:
            srm.append(m)
    
    D = gen_matrix_F(3,0.35)[1]
    D[0, 0] = 10**x
    D[1, 1] = 10**y  
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
                numpositivelocalmaxima = np.sum(nEm[argrelextrema(nEm,np.greater)]>0) > 0   #((nEm[np.where(argrelextrema(nEm, np.greater))[0]] > 0)) > 0
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
    dx = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    dy = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3] #[-3,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]
    dp_list = [(x, y) for x in dx for y in dy] 
    start_time = time.time()
    # Number of cores to use
    num_cores = 3
    #multiprocessing.cpu_count()

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_cores)
    results = pool.map(process_tuple, dp_list)
    # Close the pool to free up resources
    pool.close()
    pool.join()
    end_time = time.time()

    df_data = pd.DataFrame({'Dx': [x[0] for x in dp_list], 'Dy': [x[1] for x in dp_list], 'Percentage': results})
    df_data.to_csv(os.path.join('/Users/aibekk99/Desktop/repository/data/', 'd3.csv'), index=False)
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")