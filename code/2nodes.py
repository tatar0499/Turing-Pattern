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
import multiprocessing as mp
from scipy.signal import argrelextrema
from sympy.utilities import lambdify

# 2-node network: parameter sampling and LSA performed within one script
# command line to execute the script (example): python3 2node.py -N 10000 --fork 4  --dimensions 2 

 
ARGPARSER = argparse.ArgumentParser()

ARGPARSER.add_argument('-N', '--num_matrices', help='number of matrices to generate')
ARGPARSER.add_argument('--fork', help='number of cores to distribute workload onto')
ARGPARSER.add_argument('--dimensions', help='matrix dimensions')
#ARGPARSER.add_argument('-o', '--output', help='csv file') # new


arguments = ARGPARSER.parse_args()

N: int = int(arguments.num_matrices) if arguments.num_matrices != None else False
forks: int = int(arguments.fork) if arguments.fork != None else 1
dimensions: int = int(arguments.dimensions) if arguments.dimensions != None else False
#output_path: str = str(arguments.output) if arguments.output != None else False


if N == False:
    raise ValueError('Specify an INTEGER number of matrices to compute.')

if dimensions == False:
    raise ValueError('Specify an INTEGER number of the matrix dimensions to compute.')

def worker_func(num_matrices: int, task_tracker, lock):
    
    for _ in range(0,num_matrices):
        with lock:
            task_tracker.value += 1
            print(f'{task_tracker.value} matrices processed \r', end='')
            
    return 1

def tr_analysis(m: int, order: int, task_tracker, lock):
    if order == 2:
        sampler = qmc.LatinHypercube(d=5)
        lhs_samples = sampler.random(n=m)
        lhs_samples[:, :3] = lhs_samples[:, :3] * 99.9 + 0.1    
        lhs_samples[:, 3:5] = lhs_samples[:, 3:5] * 0.99 + 0.01   

        def equations(x, *params):
            k_uu, k_uv, k_vu, mu_u, mu_v = params
            fu = 100 * (1/(1 + (k_uu/x[0]) ** 2)) * (1/(1 + (x[1] /k_vu) ** 2)) + 0.1 - mu_u * x[0]
            fv = 100 * (1/(1 + (k_uv/x[0]) ** 2)) + 0.1 - mu_v * x[1]
            return [fu, fv] 
        roots = []

        filtered_sets = []      
        for i in range(len(lhs_samples)):
            params = tuple(lhs_samples[i])
            root = optimize.root(equations, [100,100], method='hybr', args=(params))
            res = np.all(root.x >= 0)
            val = np.around(root.x, decimals=3)
            if res == True and root.success == True and val.tolist() not in roots:
                roots.append(root.x.tolist())
                filtered_sets.append(lhs_samples[i])

        u, v, k_uu, k_vu, k_uv, mu_u, mu_v = sp.symbols('u v k_uu k_vu k_uv mu_u mu_v')
        f = 100 * (1/(1 + (k_uu/u) ** 2)) * (1/(1 + (v /k_vu) ** 2)) + 0.1 - mu_u * u
        g = 100 * (1/(1 + (k_uv/u) ** 2)) + 0.1 - mu_v * v
        f_u = f.diff(u)
        f_v = f.diff(v)
        g_u = g.diff(u)
        g_v = g.diff(v)

        stable = []
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
        turing_matrices = []

        for i in range(len(filtered_sets)):
            params = tuple(filtered_sets[i])
            param_values = {"k_uu":params[0], "k_uv":params[1], "k_vu":params[2], "mu_u":params[3], "mu_v":params[4]}
            ss = roots[i]
            param_values[u] = ss[0]
            param_values[v] = ss[1]
            f_u_value = f_u.subs(param_values)
            f_v_value = f_v.subs(param_values)
            g_u_value = g_u.subs(param_values)
            g_v_value = g_v.subs(param_values)
            J = np.array([[f_u_value, f_v_value], [g_u_value, g_v_value]])
    
            eigenvalues, eigenvectors = np.linalg.eig(J.astype(float))
            is_stable = all(eigenvalues.real < 0)
            if is_stable == False:
                with lock:
                    task_tracker.value += 1
                    print(f'{task_tracker.value} matrices processed \r', end='')
                continue
            elif is_stable == True:
                stable.append(J)


            k = np.arange(0, 101, 0.2) 
            D = np.diag([1,100])

                    
            Emi = []
            Em = []
            j = J
            for i in range(len(k)):
                R = j - D * (k[i] ** 2)
                eigval = np.linalg.eigvals(R.astype(float))
                Em.append(np.max(np.real(eigval)))
                idx_max = np.argmax(np.real(eigval))
                Emi.append(np.imag(eigval[idx_max]))
            a = np.max(Em)
            index = np.argmax(Em)
            nEm = np.array(Em)    
            if a > 0:
                if Emi[index] == 0:
                    numZeroCrossing = np.count_nonzero(np.diff(np.sign(Em)))
                    numpositivelocalmaxima = np.sum(nEm[argrelextrema(nEm,np.greater)]>0) > 0 
                    if numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 0:
                        t1a.append(j)
                        turing_matrices.append(j)
                        t1ar.append(Em)
                        t1ai.append(Emi)
                    elif numpositivelocalmaxima > 0 and numZeroCrossing == 1:
                        t1b.append(j)
                        turing_matrices.append(j)
                        t1br.append(Em)
                        t1bi.append(Emi)
                    elif numpositivelocalmaxima == 0 and numZeroCrossing % 2 == 1:
                        t2a.append(j)
                        t2ar.append(Em)
                        t2ai.append(Emi)
                    elif numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 1:
                        t2b.append(j)
                        t2br.append(Em)
                        t2bi.append(Emi)
        
            with lock:
                        task_tracker.value += 1
                        print(f'{task_tracker.value} matrices processed \r', end='')

    tr1a = len(t1a)
    tr1b = len(t1b)
    tr2a = len(t2a)
    tr2b = len(t2b)
  
    first_elements = []
    second_elements = []
    third_elements = []
    fourth_elements = []
    for matrix in turing_matrices:
        first_elements.append(float(matrix[0, 0]))
        second_elements.append(float(matrix[0, 1]))
        third_elements.append(float(matrix[1, 0]))
        fourth_elements.append(float(matrix[1, 1]))
    

    return  tr1a, tr1b, tr2a, tr2b, first_elements, second_elements, third_elements, fourth_elements

if __name__ == '__main__':
    
    # Use Manager to create shared state
    manager = mp.Manager()
    numMatricesProcessed = manager.Value('i', 0)
    lock = manager.Lock() # provide GIL

    if forks > 1:
        print(f'Multiprocessing enabled, dispatching work to {forks} cores.')
    
    num_matrices_per_worker = N // forks
    
    print(f'{num_matrices_per_worker} matrices per worker thread.')
    
    results = []
    
    with mp.Pool(processes=forks) as pool:
        for n in range(0,forks):
            result = pool.apply_async(tr_analysis, args=(num_matrices_per_worker, dimensions, numMatricesProcessed, lock))
            results.append(result)
        
        pool.close()
        pool.join()
    
    resultValues = [worker_result.get() for worker_result in results]
    print()
    
    turing_1a = sum([threadResult[0] for threadResult in resultValues])
    turing_1b = sum([threadResult[1] for threadResult in resultValues])
    turing_2a = sum([threadResult[2] for threadResult in resultValues])
    turing_2b = sum([threadResult[3] for threadResult in resultValues])


    # tr
    fe1 = []
    fe1.extend([threadResult[4] for threadResult in resultValues])
    fe_1 = [item for sublist in fe1 for item in sublist]
    fe2 = []
    fe2.extend([threadResult[5] for threadResult in resultValues])
    fe_2 = [item for sublist in fe2 for item in sublist]
    fe3 = []
    fe3.extend([threadResult[6] for threadResult in resultValues])
    fe_3 = [item for sublist in fe3 for item in sublist]
    fe4 = []
    fe4.extend([threadResult[7] for threadResult in resultValues])
    fe_4 = [item for sublist in fe4 for item in sublist]
  

    tr_data = pd.DataFrame([fe_1, fe_2, fe_3, fe_4])
    tr_data = tr_data.transpose() 
    tr_data.columns = ['First Entry', 'Second Entry', 'Third Entry', 'Fourth Entry']
    tr_data.to_csv(os.path.join('./', '2x2tr.csv'), index=False)

    print(f'{turing_1a} turing 1a instabilities')
    print(f'{turing_1b} turing 1b instabilities')
    print(f'{turing_2a} turing 2a instabilities')
    print(f'{turing_2b} turing 2b instabilities')