#!/usr/bin/env python
# coding: utf-8

# import 

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
from itertools import product
from matplotlib.lines import Line2D
import random
import scipy.stats as stats
from tqdm import tqdm
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import simps


# function:


# build matrix containing information about all the interactions on the network, including weights
# A is the full communication tensor Na No x Na No in dimension
def build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa):
    return np.kron(np.eye(Na), alpha * np.eye(No) + beta * Ao) + np.kron(Aa, gamma * np.eye(No) + delta * Ao)

def u_gauss(mu, dev, Na):
    u = np.zeros(Na)

    x = 0
    i = 0

    while i < Na:
        x = np.random.normal(mu, dev)
        if x > 0:
            u[i] = x
            i = i + 1
        else:
            i = i

    print(u)
    return u


# function for creating a vector with only positive u and with mean = mu of the gaussian:
def u_gauss2(mu, dev, Na):
    u = np.zeros(Na)

    x = 0
    i = 0
    while i < Na:
        x = np.random.normal(mu, dev)
        if x > 0:
            u[i] = x
            i = i + 1
        else:
            i = i

    mean_diff = np.mean(u) - mu
    u = u - mean_diff

    if np.all(u > 0):
        return u

    return u_gauss2(mu, dev, Na)


# previous model
#def zdot_net1(t, z, Na, No, d, A, u1, u2, b):
    # Return an Error if Na and No don't match the dimensions of z
    #if Na * No != len(z):
        #return "Na, No must match dimensions of z"

    # Define same-option and inter-option saturation functions
    #def S1(z):
        #return np.tanh(z)

    # def S2(z):
    # return 0.5 * np.tanh(2*z)

    # assemble intra-agent matrix (1)
    #inds = np.kron(np.ones((Na, Na)), np.eye(No)) > 0
    #A_SO = np.zeros((Na * No, Na * No))
    #A_SO[inds] = A[inds]
    #A_SO = A_SO

    # assemble same-option interactions and bias (2)
    #F = (np.dot(A_SO, z))
    #F = F / Na

    #F2 = np.zeros_like(F)
    # assemble inter-option interactions about option j (3)
    #for j in range(No):
        #A_j = np.zeros((Na * No, Na * No))
        #ind_mat = np.zeros((No, No))
        #ind_mat[:, j] = np.ones(No)
        #ind_mat[j, j] = 0
        #ind_mat2 = np.kron(np.ones((Na, Na)), ind_mat)
        #A_j[ind_mat2 > 0] = A[ind_mat2 > 0]
        #F2 += np.dot(A_j, z)
    #F2 = F2 / Na

    #F = F + F2
    # F = -d*z + S1(np.multiply(u1,F) + np.multiply(u2,F2)+ b)
    #F = -d * z + S1(np.multiply(u1, F) + b)
    #F = F * (1.0 / u1)

    # variation of state vector z respect time
    #proj = np.eye(Na * No)
    #dz = np.dot(proj, F)
    #return dz

# model similar to anastasia's one    
def f_64(t):
    return np.array(t).astype(np.float64)

def zdot_net1(t, z, Na, No, d, A, u1, u2, b):
    # Return an Error if Na and No don't match the dimensions of z
    if Na * No != len(z):
        return "Na, No must match dimensions of z"

    # Define same-option and inter-option saturation functions
    def S1(z):
        return np.tanh(z)

    # assemble intra-agent matrix (1)
    z = f_64(z)
    A = f_64(A)
    
    inds = np.kron(np.ones((Na, Na)), np.eye(No)) > 0
    A_SO = np.zeros((Na * No, Na * No))
    A_SO = f_64(A_SO)
    A_SO[inds] = A[inds]
    A_SO = A_SO   
    
    # assemble same-option interactions and bias (2)
    F = (np.dot(A_SO, z))

    F2 = np.zeros_like(F)
    F2 = f_64(F2)
    # assemble inter-option interactions about option j (3)
    for j in range(No):
        A_j = np.zeros((Na * No, Na * No))
        ind_mat = np.zeros((No, No))
        ind_mat[:, j] = np.ones(No)
        ind_mat[j, j] = 0
        ind_mat2 = np.kron(np.ones((Na, Na)), ind_mat)
        A_j[ind_mat2 > 0] = A[ind_mat2 > 0]
        F2 += np.dot(A_j, z)

    F = F + F2
    F = F / Na
    F = - d * z + S1(np.multiply(u1, F) + b)
    F = F*( 1.0/tau)

    # variation of state vector z respect time
    dz = F
    return dz


# riscrivere funzioni affinchè lo switching rate sia parametro della simulazione! 

#generate bias vector with tot percent of people with value != 0 on both task:
def generate_bias_vector_m(Na, No, percent, value, even=True):
    
    num_ones = int(Na * percent / 100)
    
    
    bias_vector = np.zeros(Na * No)
    
    
    indices = np.arange(Na * No)
    indicesminus = np.arange(Na * No)
    if even:
        indices = indices[indices % 2 == 0]  
        indicesminus = indicesminus[indicesminus % 2 != 0]
    else:
        indices = indices[indices % 2 != 0]  
        indicesminus = indicesminus[indicesminus % 2 == 0]
    
    bias_vector[indices[:num_ones]] = value
    bias_vector[indicesminus[:num_ones]] = -value
    
    return bias_vector

#funzione per trovare time window corretta
def find_indices_between_elements(arr, t):
    # Inizializza gli indici a None
    first_index = None
    second_index = None

    for i, element in enumerate(arr):
        if element <= t:
            # Trova l'indice più vicino a t
            first_index = i
        else:
            # L'elemento successivo è maggiore di t, quindi usciamo dal ciclo
            break

    if first_index is not None:
        # Calcola l'indice successivo a first_index
        second_index = first_index + 1

    return first_index, second_index

#funzione per creare vettore thresholds recolari
def find_thresh(T, rate):
    
    times = [0.0, 200.0]
    Tt = T - 200
    
    nb = int(rate*Tt) #nb of task switches
        
    #regular time interval:
    t_int = Tt / nb
    
    t_i = 200.0
    for j in range(nb):
        t_i += t_int
        times.append(t_i)
    
    return times

#funzione per creare vettore thresholds con t poissoniani
def find_thresh_p(T, rate):
    
    time_thresh = [0.0, 200.0]
    
    Tt = T - 200
    nb = int(rate*Tt) 
    
    # Poisson process for bias time intervals
    t = 200.0
    while t < T:
        t += np.random.exponential(1.0/rate)  # Generate time intervals according to Poisson process
        if t <= T:
            time_thresh.append(t)
    
    if len(time_thresh)-1 == nb:
        return time_thresh
    else:
        return find_thresh_p(T, rate)


#funzione con biases a t regolari

def sim_net1(z, T, Na, No, d, A, u1, u2, rate, percent, value, times):
    
    b = []
    
    Tt = T - 200
    nb = int(rate*Tt) #nb of task switches
    
    #definition of the biases:
    for i in range(nb + 1):
        
        if i == 0:
            b_i = np.zeros(Na*No) * bias_w
        elif i != 0 and i % 2 != 0:
            b_i = generate_bias_vector_m(Na, No, percent, value, even=True) * bias_w
        elif i != 0 and i % 2 == 0:
            b_i = generate_bias_vector_m(Na, No, percent, value, even=False) * bias_w
        
        b.append(b_i)
    
    def f(t, z):
    
        if t != T:
            fi, se = find_indices_between_elements(times, t)
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b[fi])
            
        if t == T:
            fi, se = find_indices_between_elements(times, t)
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b[fi-1])
    
    zs = solve_ivp(f, [0, T], z, dense_output=False, t_eval=np.linspace(0, T, 200), max_step=1000, method="Radau")
    
    return zs


#funzione con biases ad intervalli regolari e con calcolo intervalli interno
def sim_net11(z, T, Na, No, d, A, u1, u2, rate, percent, value):
    
    b = []
    time_thresh = [0.0, 200.0]
    Tt = T - 200
    
    nb = int(rate*Tt) #nb of task switches
    
    #definition of the biases:
    for i in range(nb + 1):
        
        if i == 0:
            b_i = np.zeros(Na*No) * bias_w
        elif i != 0 and i % 2 != 0:
            b_i = generate_bias_vector_m(Na, No, percent, value, even=True) * bias_w
        elif i != 0 and i % 2 == 0:
            b_i = generate_bias_vector_m(Na, No, percent, value, even=False) * bias_w
        
        b.append(b_i)
        
    #regular time interval:
    t_int = Tt / nb
    
    t_i = 200.0
    for j in range(nb):
        t_i += t_int
        time_thresh.append(t_i)
    
    #run simulation
    def f(t, z):
    
        if t != T:
            fi, se = find_indices_between_elements(time_thresh, t)
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b[fi])
            
        if t == T:
            fi, se = find_indices_between_elements(time_thresh, t)
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b[fi-1])
            
    zs = solve_ivp(f, [0, T], z, dense_output=True, t_eval=np.linspace(0, T, 200), max_step=1000)

    return zs

# vecchia funzione di simulazione
def sim_net0(z, T, Na, No, d, A, u1, u2, b1, b2, b3, b4, time_threshold1, time_threshold2, time_threshold3):
    global debug, times
    debug = []
    times = []

    # run simulation
    def f(t, z):
        if t < time_threshold1:
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b1)
        elif time_threshold1 <= t < time_threshold2:
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b2)
        elif time_threshold2 <= t < time_threshold3:
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b3)
        elif t >= time_threshold3:
            return zdot_net1(t, z, Na, No, d, A, u1, u2, b4)

    zs = solve_ivp(f, [0, T], z, dense_output=True, t_eval=np.linspace(0, T, 200), max_step=1000)

    return zs


# function to pass from a vector Na*No to a vector of Na
def new_zs(zs, Na, No):
    n2 = zs.shape[1]
    zsn = np.zeros((Na, n2))
    j = i = 0
    for i in range(0, Na * No, 2):
        zsn[j] = (zs[i] - zs[i + 1]) / 2.0
        j = j + 1

    return zsn

# evaluation of the performance function:

#performance on the aerea
def performance(x, y, x_min, x_max, sopra_zero=True):
    """
    Calcola l'area sopra o sotto la curva dati una lista di punti (x, y) nell'intervallo [x_min, x_max] utilizzando la regola di Simpson.
    
    :param x: Lista dei punti x
    :param y: Lista dei punti y (valori della funzione)
    :param x_min: Limite inferiore dell'intervallo
    :param x_max: Limite superiore dell'intervallo
    :param sopra_zero: Se True, calcola l'area sopra y=0, altrimenti calcola l'area sotto y=0.
    :return: L'area sopra o sotto la curva nell'intervallo specificato rispetto a y=0.
    """
    if len(x) != len(y):
        raise ValueError("Le liste x e y devono avere la stessa lunghezza")
    
    # Trova gli indici in cui x è all'interno dell'intervallo [x_min, x_max]
    mask = (x >= x_min) & (x <= x_max)
    x_interval = x[mask]
    y_interval = y[mask]
    
    if sopra_zero:
        y_interval = np.maximum(y_interval, 0)
    else:
        y_interval = np.minimum(y_interval, 0)
    
    return simps(y_interval, x_interval)


#performance on the crossing time
def crossing_time(x, y_i, tmin, tmax):
    
    zero_crossing_times = []
    
    for i in range(y_i.shape[0]):
        y = y_i[i]
        for j in range(1, len(y)):
            if np.sign(y[j]) != np.sign(y[j - 1]):
                # Se il segno cambia tra due punti consecutivi, calcola il tempo di attraversamento
                t_crossing = x[j - 1] + (x[j] - x[j - 1]) * abs(y[j - 1]) / (abs(y[j - 1]) + abs(y[j]))
            
                # Verifica se il tempo di attraversamento è compreso nell'intervallo [tmin, tmax]
                if tmin <= t_crossing <= tmax:
                    zero_crossing_times.append(t_crossing)
    
    median_time = np.median(zero_crossing_times)
    median_crossing_time = median_time - tmin
    
    return median_crossing_time

#performance on the total crossing time (reward-penalty) -> metric 2 and 3 together
def crossing_time_tot(x, y_i, tmin, tmax):
    
    zero_crossing_times = []
    
    for i in range(y_i.shape[0]):
        y = y_i[i]
        for j in range(1, len(y)):
            if np.sign(y[j]) != np.sign(y[j - 1]):
                # Se il segno cambia tra due punti consecutivi, calcola il tempo di attraversamento
                t_crossing = x[j - 1] + (x[j] - x[j - 1]) * abs(y[j - 1]) / (abs(y[j - 1]) + abs(y[j]))
            
                # Verifica se il tempo di attraversamento è compreso nell'intervallo [tmin, tmax]
                if tmin <= t_crossing <= tmax:
                    zero_crossing_times.append(t_crossing)
    
    #zero_crossing_times = np.nan_to_num(zero_crossing_times, nan=0.0)
    #metric on the group median switching time
    median_time = np.median(zero_crossing_times)
    median_crossing_time = abs(median_time - tmin)
    
    #metric on the tot time agents are doing the right task
    m=0
    t_w = 0; t_wt = 0; t_r = 0; t_rt = 0
    t_tot = 0
    
    for m in zero_crossing_times:
        t_w = abs(m - tmin)
        t_wt += t_w
        
        t_r = abs(tmax - m) 
        t_rt += t_r
        
    t_tot = t_rt - t_wt
    
    return median_crossing_time, t_tot


#-----------------------------------------------------------------------------------------------------------------------------------------

# case 4 agents:

Na = 4  # number of agents
No = 2  # number of tasks

#null initial conditions
z0 = np.zeros(Na*No)

z = z0
z1 = z0

# simulation run time
T = 2200

# model parameters
d = 0.2
alpha = 0.02
beta = 0.01
gamma = 0.02
delta = 0.0
tau = 10
ue = 1

# Matrices
Ao = np.array([[1, -1], [-1, 1]])  # incongruent stimuli
Aa = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])  # only positive interaction and all agents connetcted

# Dynamic:

# build A
A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)

Tmin = 0
Tmax = 2200

#-----------------------------------------------------------------------------------------------------------------------------------------

# 4 BIASES:

# list to store data:
mu_4, mu_6, mu_8, mu_10, mu_12, mu_14 = [], [], [], [], [], []
cv_4, cv_6, cv_8, cv_10, cv_12, cv_14 = [], [], [], [], [], []
#dev_2, dev_4, dev_6, dev_8 = [], [], [], [], []

p_4, p_6, p_8, p_10, p_12, p_14 = [], [], [], [], [], []
mean_4, mean_6, mean_8, mean_10, mean_12, mean_14 = [], [], [], [], [], []
std_4, std_6, std_8, std_10, std_12, std_14 = [], [], [], [], [], []
diff_4, diff_6, diff_8, diff_10, diff_12, diff_14 = [], [], [], [], [], []

p2_4, p2_6, p2_8, p2_10, p2_12, p2_14 = [], [], [], [], [], []
mean2_4, mean2_6, mean2_8, mean2_10, mean2_12, mean2_14 = [], [], [], [], [], []
std2_4, std2_6, std2_8, std2_10, std2_12, std2_14 = [], [], [], [], [], []
diff2_4, diff2_6, diff2_8, diff2_10, diff2_12, diff2_14 = [], [], [], [], [], []

p3_4, p3_6, p3_8, p3_10, p3_12, p3_14 = [], [], [], [], [], []
mean3_4, mean3_6, mean3_8, mean3_10, mean3_12, mean3_14 = [], [], [], [], [], []
std3_4, std3_6, std3_8, std3_10, std3_12, std3_14 = [], [], [], [], [], []
diff3_4, diff3_6, diff3_8, diff3_10, diff3_12, diff3_14 = [], [], [], [], [], []

Ahom_4, Ahom_6, Ahom_8, Ahom_10, Ahom_12, Ahom_14 = [], [], [], [], [], []
t_homs_4, t_homs_6, t_homs_8, t_homs_10, t_homs_12, t_homs_14 = [], [], [], [], [], []
T_homs_4, T_homs_6, T_homs_8, T_homs_10, T_homs_12, T_homs_14 = [], [], [], [], [], []

u_4, u_6, u_8, u_10, u_12, u_14 = [], [], [], [], [], []
bet_4, bet_6, bet_8, bet_10, bet_12, bet_14 = [], [], [], [], [], []
area_4, area_6, area_8, area_10, area_12, area_14 = [], [], [], [], [], []
mus_4, mus_6, mus_8, mus_10, mus_12, mus_14 = [], [], [], [], [], []
cvs_4, cvs_6, cvs_8, cvs_10, cvs_12, cvs_14 = [], [], [], [], [], []
area_hom_4, area_hom_6, area_hom_8, area_hom_10, area_hom_12, area_hom_14 = [], [], [], [], [], []


bias_w = 0.1
value = 1
percent = 100

# 4 TASK SWITCHES

T = 2200
rate = 0.002

times4 = find_thresh(T, rate)

def hom_4(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_4.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_4.append(t_hom)
    T_homs_4.append(T_hom)

    return Atot, t_hom, T_hom



def het_4(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_4.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_4.append(Atot2)
        area_hom_4.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_4.append(1)
        
        else: bet_4.append(-1)
        
        mus_4.append(mu)
        cvs_4.append(cv)
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett

       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1

    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_4.append(p)
    mu_4.append(mu)
    cv_4.append(cv)
    mean_4.append(mean)
    std_4.append(std)
    diff_4.append(diff)
    
    p2_4.append(p2)
    mean2_4.append(mean2)
    std2_4.append(std2)
    diff2_4.append(diff2)
    
    p3_4.append(p3)
    mean3_4.append(mean3)
    std3_4.append(std3)
    diff3_4.append(diff3)



def perf_4(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_4(mu, pe, times)
    het_4(mu, dev, pe, Atot, t_hom, T_hom, times)

# SIMULATION:

mus = [6.176923]
#mus = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
#devs = [4.0, 3.0, 2.0, 1.0]
cvs = [0.25, 0.50, 0.75, 1.0]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_4(mu, cv*mu, 1.0, times4)
        
#mu = 4.8   
#for cv in tqdm(cvs):
    #perf_4(mu, cv*mu, pe=1.0, tmin1 = 500, tmax1 = 800, tmin2 = 800, tmax2 = 1100, tmin3 = 1100, tmax3 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_4, 'cv': cv_4, 'performance1_hom': Ahom_4, 'performance1_het%': p_4, 'performance1_het_mean': mean_4,
        'performance1_het_dev': std_4, 'performance1_diff': diff_4, 'performance2_hom': t_homs_4, 'performance2_het%': p2_4,
        'performance2_het_mean': mean2_4, 'performance2_het_dev': std2_4, 'performance2_diff': diff2_4, 'performance3_hom': T_homs_4,               'performance3_het%': p3_4, 'performance3_het_mean': mean3_4, 'performance3_het_dev': std3_4, 'performance3_diff': diff3_4}
                                        
df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_4, 'cv': cvs_4, 'u_vector': u_4, 'perf het > hom?': bet_4, 'performance_hom': area_hom_4, 
         'performance1_het_dev': area_4}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_4_biases_distr.csv', index=False)


# 6 TASK SWITCHES

T = 2200
rate = 0.003

times6 = find_thresh(T, rate)

def hom_6(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_6.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_6.append(t_hom)
    T_homs_6.append(T_hom)

    return Atot, t_hom, T_hom



def het_6(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_6.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_6.append(Atot2)
        area_hom_6.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_6.append(1)
        
        else: bet_6.append(-1)
        
        mus_6.append(mu)
        cvs_6.append(cv)
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett

       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1

    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_6.append(p)
    mu_6.append(mu)
    cv_6.append(cv)
    mean_6.append(mean)
    std_6.append(std)
    diff_6.append(diff)
    
    p2_6.append(p2)
    mean2_6.append(mean2)
    std2_6.append(std2)
    diff2_6.append(diff2)
    
    p3_6.append(p3)
    mean3_6.append(mean3)
    std3_6.append(std3)
    diff3_6.append(diff3)



def perf_6(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_6(mu, pe, times)
    het_6(mu, dev, pe, Atot, t_hom, T_hom, times)


# SIMULATION:

mus = [4.758974]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_6(mu, cv*mu, 1.0, times6)

        
#mu = 2.9 
#for cv in tqdm(cvs):
    #perf_6(mu, cv*mu, pe=1.0, tmin1 = 400, tmax1 = 600, tmin2 = 600, tmax2 = 800, tmin3 = 800, tmax3 = 1000, tmin4 = 1000, tmax4 = 1200,              tmin5 = 1200, tmax5 = 1400)
    

# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_6, 'cv': cv_6, 'performance1_hom': Ahom_6, 'performance1_het%': p_6, 'performance1_het_mean': mean_6,
        'performance1_het_dev': std_6, 'performance1_diff': diff_6, 'performance2_hom': t_homs_6, 'performance2_het%': p2_6,
        'performance2_het_mean': mean2_6, 'performance2_het_dev': std2_6, 'performance2_diff': diff2_6, 'performance3_hom': T_homs_6,               'performance3_het%': p3_6, 'performance3_het_mean': mean3_6, 'performance3_het_dev': std3_6, 'performance3_diff': diff3_6}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_6, 'cv': cvs_6, 'u_vector': u_6, 'perf het > hom?': bet_6, 'performance_hom': area_hom_6, 
         'performance1_het_dev': area_6}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_4_biases_distr.csv', index=False)


# 8 TASK SWITCHES:

T = 2200
rate = 0.004

times8 = find_thresh(T, rate)

def hom_8(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_8.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_8.append(t_hom)
    T_homs_8.append(T_hom)

    return Atot, t_hom, T_hom



def het_8(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_8.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_8.append(Atot2)
        area_hom_8.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_8.append(1)
        
        else: bet_8.append(-1)
        
        mus_8.append(mu)
        cvs_8.append(cv)
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett

       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
        
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1


    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_8.append(p)
    mu_8.append(mu)
    cv_8.append(cv)
    mean_8.append(mean)
    std_8.append(std)
    diff_8.append(diff)
    
    p2_8.append(p2)
    mean2_8.append(mean2)
    std2_8.append(std2)
    diff2_8.append(diff2)
    
    p3_8.append(p3)
    mean3_8.append(mean3)
    std3_8.append(std3)
    diff3_8.append(diff3)


def perf_8(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_8(mu, pe, times)
    het_8(mu, dev, pe, Atot, t_hom, T_hom, times)
    

# SIMULATION:

mus = [3.543590]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_8(mu, cv*mu, 1.0, times8)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_8, 'cv': cv_8, 'performance1_hom': Ahom_8, 'performance1_het%': p_8, 'performance1_het_mean': mean_8,
        'performance1_het_dev': std_8, 'performance1_diff': diff_8, 'performance2_hom': t_homs_8, 'performance2_het%': p2_8,
        'performance2_het_mean': mean2_8, 'performance2_het_dev': std2_8, 'performance2_diff': diff2_8, 'performance3_hom': T_homs_8,               'performance3_het%': p3_8, 'performance3_het_mean': mean3_8, 'performance3_het_dev': std3_8, 'performance3_diff': diff3_8}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_8, 'cv': cvs_8, 'u_vector': u_8, 'perf het > hom?': bet_8, 'performance_hom': area_hom_8, 
         'performance1_het_dev': area_8}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_4_biases_distr.csv', index=False)


# 10 TASK SWITCHES:

T = 2200
rate = 0.005

times10 = find_thresh(T, rate)

def hom_10(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_10.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_10.append(t_hom)
    T_homs_10.append(T_hom)

    return Atot, t_hom, T_hom


def het_10(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_10.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_10.append(Atot2)
        area_hom_10.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_10.append(1)
        
        else: bet_10.append(-1)
        
        mus_10.append(mu)
        cvs_10.append(cv)
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett
       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1


    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_10.append(p)
    mu_10.append(mu)
    cv_10.append(cv)
    mean_10.append(mean)
    std_10.append(std)
    diff_10.append(diff)
    
    p2_10.append(p2)
    mean2_10.append(mean2)
    std2_10.append(std2)
    diff2_10.append(diff2)
    
    p3_10.append(p3)
    mean3_10.append(mean3)
    std3_10.append(std3)
    diff3_10.append(diff3)


def perf_10(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_10(mu, pe, times)
    het_10(mu, dev, pe, Atot, t_hom, T_hom, times)
    

# SIMULATION:

mus = [2.125641]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_10(mu, cv*mu, 1.0, times10)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_10, 'cv': cv_10, 'performance1_hom': Ahom_10, 'performance1_het%': p_10, 'performance1_het_mean': mean_10,
        'performance1_het_dev': std_10, 'performance1_diff': diff_10, 'performance2_hom': t_homs_10, 'performance2_het%': p2_10,
        'performance2_het_mean': mean2_10, 'performance2_het_dev': std2_10, 'performance2_diff': diff2_10, 'performance3_hom': T_homs_10,               'performance3_het%': p3_10, 'performance3_het_mean': mean3_10, 'performance3_het_dev': std3_10, 'performance3_diff': diff3_10}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_10, 'cv': cvs_10, 'u_vector': u_10, 'perf het > hom?': bet_10, 'performance_hom': area_hom_10, 
         'performance1_het_dev': area_10}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_4_biases_distr.csv', index=False)

# 12 TASK SWITCHES:

T = 2200
rate = 0.006

times12 = find_thresh(T, rate)

def hom_12(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_12.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_12.append(t_hom)
    T_homs_12.append(T_hom)

    return Atot, t_hom, T_hom



def het_12(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_12.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_12.append(Atot2)
        area_hom_12.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_12.append(1)
        
        else: bet_12.append(-1)
        
        mus_12.append(mu)
        cvs_12.append(cv)
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett

       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
        
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1


    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_12.append(p)
    mu_12.append(mu)
    cv_12.append(cv)
    mean_12.append(mean)
    std_12.append(std)
    diff_12.append(diff)
    
    p2_12.append(p2)
    mean2_12.append(mean2)
    std2_12.append(std2)
    diff2_12.append(diff2)
    
    p3_12.append(p3)
    mean3_12.append(mean3)
    std3_12.append(std3)
    diff3_12.append(diff3)


def perf_12(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_12(mu, pe, times)
    het_12(mu, dev, pe, Atot, t_hom, T_hom, times)
    

# SIMULATION:

mus = [0.910256]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_12(mu, cv*mu, 1.0, times12)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_12, 'cv': cv_12, 'performance1_hom': Ahom_12, 'performance1_het%': p_12, 'performance1_het_mean': mean_12,
        'performance1_het_dev': std_12, 'performance1_diff': diff_12, 'performance2_hom': t_homs_12, 'performance2_het%': p2_12,
        'performance2_het_mean': mean2_12, 'performance2_het_dev': std2_12, 'performance2_diff': diff2_12, 'performance3_hom': T_homs_12,               'performance3_het%': p3_12, 'performance3_het_mean': mean3_12, 'performance3_het_dev': std3_12, 'performance3_diff': diff3_12}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_12, 'cv': cvs_12, 'u_vector': u_12, 'perf het > hom?': bet_12, 'performance_hom': area_hom_12, 
         'performance1_het_dev': area_12}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_4_biases_distr.csv', index=False)


# 14 TASK SWITCHES:

T = 2200
rate = 0.007

times14 = find_thresh(T, rate)

def hom_14(mu, pe, times):
    # simulation
    u = mu
    A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
    z = z1
    zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
    zs.y = new_zs(zs.y, Na, No)

    # metric 1
    Ar = A1 = A11 = At = Att = Aw = Atot = 0

    for i in range(0, Na):
        At = Att = 0
        for j in range(1, len(times)-1):
            if j % 2 != 0:
                A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
            else:
                A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))
       
            At += A1
        Ar += At
        
        for k in range(1, len(times)-1):
            if k % 2 != 0:
                A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
            else:
                A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)
            
            Att += A11
        Aw += Att

    Atot = Ar - (Aw * pe)
    
    Ahom_14.append(Atot)
    
    # metric 2 + 3
    t_hom = 0
    T_hom = 0
    
    for j in range(2, len(times)-1):
        t_homt, T_homt = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
        t_hom += t_homt 
        T_hom += T_homt

    t_homs_14.append(t_hom)
    T_homs_14.append(T_hom)

    return Atot, t_hom, T_hom



def het_14(mu, dev, pe, Atot, t_hom, T_hom, times):
    n = 300
    p = 0
    count = 0
    p2 = 0
    count2 = 0
    p3 = 0
    count3 = 0
    Atot2s = []
    t_hets = []
    T_hets = []

    for j in range(n):
        Ar = A1 = A11 = At = Att = Aw = Atot2 = 0
        u = u_gauss2(mu, dev, Na)
        u = np.array(u).repeat(2)
        u_14.append(u)
        A = build_A(Na, No, alpha, beta, gamma, delta, Ao, Aa)
        z = z1
        zs = sim_net1(z, T, Na, No, d, A, u, ue, rate, percent, value, times)
        zs.y = new_zs(zs.y, Na, No)
        
        # metric 1
        for i in range(0, Na):
            At = Att = 0
            for j in range(1, len(times)-1):
                if j % 2 != 0:
                    A1 = performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=True)
                else:
                    A1 = abs(performance(zs.t, zs.y[i], x_min=times[j], x_max=times[j+1], sopra_zero=False))

                At += A1
            Ar += At

            for k in range(1, len(times)-1):
                if k % 2 != 0:
                    A11 = abs(performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=False))
                else:
                    A11 = performance(zs.t, zs.y[i], x_min=times[k], x_max=times[k+1], sopra_zero=True)

                Att += A11
            Aw += Att

        Atot2 = Ar - (Aw * pe)
        Atot2s.append(Atot2)
        area_14.append(Atot2)
        area_hom_14.append(Atot)

        if Atot2 > Atot:
            count = count + 1
            bet_14.append(1)
        
        else: bet_14.append(-1)
        
        mus_14.append(mu)
        cvs_14.append(cv)

        if Atot2 > Atot:
            count = count + 1
            
        # metric 2 + 3
        t_het = 0
        T_het = 0
        
        for j in range(2, len(times)-1):
            t_hett, T_hett = crossing_time_tot(zs.t, zs.y, times[j], times[j+1])
            t_het += t_hett 
            T_het += T_hett

       
        t_hets.append(t_het)
        T_hets.append(T_het)
        
    
        if t_het < t_hom:
            count2 = count2 + 1
            
        if T_het > T_hom:
            count3 = count3 + 1


    p = (count / n) * 100
    p2 = (count2 / n) * 100
    p3 = (count3 / n) * 100

    mean = np.mean(Atot2s)
    std = np.std(Atot2s)
    diff = np.abs(Atot - mean)
    
    mean2 = np.mean(t_hets)
    std2 = np.std(t_hets)
    diff2 = np.abs(t_hom - mean2)
    
    mean3 = np.mean(T_hets)
    std3 = np.std(T_hets)
    diff3 = np.abs(T_hom - mean3)

    p_14.append(p)
    mu_14.append(mu)
    cv_14.append(cv)
    mean_14.append(mean)
    std_14.append(std)
    diff_14.append(diff)
    
    p2_14.append(p2)
    mean2_14.append(mean2)
    std2_14.append(std2)
    diff2_14.append(diff2)
    
    p3_14.append(p3)
    mean3_14.append(mean3)
    std3_14.append(std3)
    diff3_14.append(diff3)


def perf_14(mu, dev, pe, times):
    Atot, t_hom, T_hom = hom_14(mu, pe, times)
    het_14(mu, dev, pe, Atot, t_hom, T_hom, times)
    

# SIMULATION:

mus = [0.1]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_14(mu, cv*mu, 1.0, times14)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_14, 'cv': cv_14, 'performance1_hom': Ahom_14, 'performance1_het%': p_14, 'performance1_het_mean': mean_14,
        'performance1_het_dev': std_14, 'performance1_diff': diff_14, 'performance2_hom': t_homs_14, 'performance2_het%': p2_14,
        'performance2_het_mean': mean2_14, 'performance2_het_dev': std2_14, 'performance2_diff': diff2_14, 'performance3_hom': T_homs_14,               'performance3_het%': p3_14, 'performance3_het_mean': mean3_14, 'performance3_het_dev': std3_14, 'performance3_diff': diff3_14}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_4_biases.csv', index=False)

#Data2
data1 = {'mu': mus_14, 'cv': cvs_14, 'u_vector': u_14, 'perf het > hom?': bet_14, 'performance_hom': area_hom_14, 
         'performance1_het_dev': area_14}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_4_biases_distr.csv', index=False)


#--------------------------------------------------------------------------------------------------------------------------------------
#3 BIASES
# list to store data:

mu_4, mu_6, mu_8, mu_10, mu_12, mu_14 = [], [], [], [], [], []
cv_4, cv_6, cv_8, cv_10, cv_12, cv_14 = [], [], [], [], [], []
#dev_2, dev_4, dev_6, dev_8 = [], [], [], [], []

p_4, p_6, p_8, p_10, p_12, p_14 = [], [], [], [], [], []
mean_4, mean_6, mean_8, mean_10, mean_12, mean_14 = [], [], [], [], [], []
std_4, std_6, std_8, std_10, std_12, std_14 = [], [], [], [], [], []
diff_4, diff_6, diff_8, diff_10, diff_12, diff_14 = [], [], [], [], [], []

p2_4, p2_6, p2_8, p2_10, p2_12, p2_14 = [], [], [], [], [], []
mean2_4, mean2_6, mean2_8, mean2_10, mean2_12, mean2_14 = [], [], [], [], [], []
std2_4, std2_6, std2_8, std2_10, std2_12, std2_14 = [], [], [], [], [], []
diff2_4, diff2_6, diff2_8, diff2_10, diff2_12, diff2_14 = [], [], [], [], [], []

p3_4, p3_6, p3_8, p3_10, p3_12, p3_14 = [], [], [], [], [], []
mean3_4, mean3_6, mean3_8, mean3_10, mean3_12, mean3_14 = [], [], [], [], [], []
std3_4, std3_6, std3_8, std3_10, std3_12, std3_14 = [], [], [], [], [], []
diff3_4, diff3_6, diff3_8, diff3_10, diff3_12, diff3_14 = [], [], [], [], [], []

Ahom_4, Ahom_6, Ahom_8, Ahom_10, Ahom_12, Ahom_14 = [], [], [], [], [], []
t_homs_4, t_homs_6, t_homs_8, t_homs_10, t_homs_12, t_homs_14 = [], [], [], [], [], []
T_homs_4, T_homs_6, T_homs_8, T_homs_10, T_homs_12, T_homs_14 = [], [], [], [], [], []

u_4, u_6, u_8, u_10, u_12, u_14 = [], [], [], [], [], []
bet_4, bet_6, bet_8, bet_10, bet_12, bet_14 = [], [], [], [], [], []
area_4, area_6, area_8, area_10, area_12, area_14 = [], [], [], [], [], []
mus_4, mus_6, mus_8, mus_10, mus_12, mus_14 = [], [], [], [], [], []
cvs_4, cvs_6, cvs_8, cvs_10, cvs_12, cvs_14 = [], [], [], [], [], []
area_hom_4, area_hom_6, area_hom_8, area_hom_10, area_hom_12, area_hom_14 = [], [], [], [], [], []

# BIASES:

bias_w = 0.1
value = 1
percent = 75

# 4 TASK SWITCHES

T = 2200
rate = 0.002

# SIMULATION:

mus = [6.176923]
#mus = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
#devs = [4.0, 3.0, 2.0, 1.0]
cvs = [0.25, 0.50, 0.75, 1.0]


for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_4(mu, cv*mu, 1.0, times4)
        
#mu = 4.8   
#for cv in tqdm(cvs):
    #perf_4(mu, cv*mu, pe=1.0, tmin1 = 500, tmax1 = 800, tmin2 = 800, tmax2 = 1100, tmin3 = 1100, tmax3 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_4, 'cv': cv_4, 'performance1_hom': Ahom_4, 'performance1_het%': p_4, 'performance1_het_mean': mean_4,
        'performance1_het_dev': std_4, 'performance1_diff': diff_4, 'performance2_hom': t_homs_4, 'performance2_het%': p2_4,
        'performance2_het_mean': mean2_4, 'performance2_het_dev': std2_4, 'performance2_diff': diff2_4, 'performance3_hom': T_homs_4,               'performance3_het%': p3_4, 'performance3_het_mean': mean3_4, 'performance3_het_dev': std3_4, 'performance3_diff': diff3_4}
                                        
df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_4, 'cv': cvs_4, 'u_vector': u_4, 'perf het > hom?': bet_4, 'performance_hom': area_hom_4, 
         'performance1_het_dev': area_4}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_3_biases_distr.csv', index=False)


# 6 TASK SWITCHES

T = 2200
rate = 0.003

# SIMULATION:

mus = [4.758974]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_6(mu, cv*mu, 1.0, times6)

        
#mu = 2.9 
#for cv in tqdm(cvs):
    #perf_6(mu, cv*mu, pe=1.0, tmin1 = 400, tmax1 = 600, tmin2 = 600, tmax2 = 800, tmin3 = 800, tmax3 = 1000, tmin4 = 1000, tmax4 = 1200,              tmin5 = 1200, tmax5 = 1400)
    

# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_6, 'cv': cv_6, 'performance1_hom': Ahom_6, 'performance1_het%': p_6, 'performance1_het_mean': mean_6,
        'performance1_het_dev': std_6, 'performance1_diff': diff_6, 'performance2_hom': t_homs_6, 'performance2_het%': p2_6,
        'performance2_het_mean': mean2_6, 'performance2_het_dev': std2_6, 'performance2_diff': diff2_6, 'performance3_hom': T_homs_6,               'performance3_het%': p3_6, 'performance3_het_mean': mean3_6, 'performance3_het_dev': std3_6, 'performance3_diff': diff3_6}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_6, 'cv': cvs_6, 'u_vector': u_6, 'perf het > hom?': bet_6, 'performance_hom': area_hom_6, 
         'performance1_het_dev': area_6}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_3_biases_distr.csv', index=False)


# 8 TASK SWITCHES:

T = 2200
rate = 0.004

# SIMULATION:

mus = [3.543590]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_8(mu, cv*mu, 1.0, times8)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_8, 'cv': cv_8, 'performance1_hom': Ahom_8, 'performance1_het%': p_8, 'performance1_het_mean': mean_8,
        'performance1_het_dev': std_8, 'performance1_diff': diff_8, 'performance2_hom': t_homs_8, 'performance2_het%': p2_8,
        'performance2_het_mean': mean2_8, 'performance2_het_dev': std2_8, 'performance2_diff': diff2_8, 'performance3_hom': T_homs_8,               'performance3_het%': p3_8, 'performance3_het_mean': mean3_8, 'performance3_het_dev': std3_8, 'performance3_diff': diff3_8}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_8, 'cv': cvs_8, 'u_vector': u_8, 'perf het > hom?': bet_8, 'performance_hom': area_hom_8, 
         'performance1_het_dev': area_8}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_3_biases_distr.csv', index=False)


# 10 TASK SWITCHES:

T = 2200
rate = 0.005

# SIMULATION:
mus = [2.328205]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_10(mu, cv*mu, 1.0, times10)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_10, 'cv': cv_10, 'performance1_hom': Ahom_10, 'performance1_het%': p_10, 'performance1_het_mean': mean_10,
        'performance1_het_dev': std_10, 'performance1_diff': diff_10, 'performance2_hom': t_homs_10, 'performance2_het%': p2_10,
        'performance2_het_mean': mean2_10, 'performance2_het_dev': std2_10, 'performance2_diff': diff2_10, 'performance3_hom': T_homs_10,               'performance3_het%': p3_10, 'performance3_het_mean': mean3_10, 'performance3_het_dev': std3_10, 'performance3_diff': diff3_10}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_10, 'cv': cvs_10, 'u_vector': u_10, 'perf het > hom?': bet_10, 'performance_hom': area_hom_10, 
         'performance1_het_dev': area_10}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_3_biases_distr.csv', index=False)



# 12 TASK SWITCHES:

T = 2200
rate = 0.006

# SIMULATION:

mus = [0.910256]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_12(mu, cv*mu, 1.0, times12)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_12, 'cv': cv_12, 'performance1_hom': Ahom_12, 'performance1_het%': p_12, 'performance1_het_mean': mean_12,
        'performance1_het_dev': std_12, 'performance1_diff': diff_12, 'performance2_hom': t_homs_12, 'performance2_het%': p2_12,
        'performance2_het_mean': mean2_12, 'performance2_het_dev': std2_12, 'performance2_diff': diff2_12, 'performance3_hom': T_homs_12,               'performance3_het%': p3_12, 'performance3_het_mean': mean3_12, 'performance3_het_dev': std3_12, 'performance3_diff': diff3_12}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_12, 'cv': cvs_12, 'u_vector': u_12, 'perf het > hom?': bet_12, 'performance_hom': area_hom_12, 
         'performance1_het_dev': area_12}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_3_biases_distr.csv', index=False)

# 14 TASK SWITCHES:

T = 2200
rate = 0.007

# SIMULATION:

mus = [0.1]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_14(mu, cv*mu, 1.0, times14)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_14, 'cv': cv_14, 'performance1_hom': Ahom_14, 'performance1_het%': p_14, 'performance1_het_mean': mean_14,
        'performance1_het_dev': std_14, 'performance1_diff': diff_14, 'performance2_hom': t_homs_14, 'performance2_het%': p2_14,
        'performance2_het_mean': mean2_14, 'performance2_het_dev': std2_14, 'performance2_diff': diff2_14, 'performance3_hom': T_homs_14,               'performance3_het%': p3_14, 'performance3_het_mean': mean3_14, 'performance3_het_dev': std3_14, 'performance3_diff': diff3_14}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_3_biases.csv', index=False)

#Data2
data1 = {'mu': mus_14, 'cv': cvs_14, 'u_vector': u_14, 'perf het > hom?': bet_14, 'performance_hom': area_hom_14, 
         'performance1_het_dev': area_14}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_3_biases_distr.csv', index=False)

#--------------------------------------------------------------------------------------------------------------------------------------

#2 BIASES

# list to store data:

mu_4, mu_6, mu_8, mu_10, mu_12, mu_14 = [], [], [], [], [], []
cv_4, cv_6, cv_8, cv_10, cv_12, cv_14 = [], [], [], [], [], []
#dev_2, dev_4, dev_6, dev_8 = [], [], [], [], []

p_4, p_6, p_8, p_10, p_12, p_14 = [], [], [], [], [], []
mean_4, mean_6, mean_8, mean_10, mean_12, mean_14 = [], [], [], [], [], []
std_4, std_6, std_8, std_10, std_12, std_14 = [], [], [], [], [], []
diff_4, diff_6, diff_8, diff_10, diff_12, diff_14 = [], [], [], [], [], []

p2_4, p2_6, p2_8, p2_10, p2_12, p2_14 = [], [], [], [], [], []
mean2_4, mean2_6, mean2_8, mean2_10, mean2_12, mean2_14 = [], [], [], [], [], []
std2_4, std2_6, std2_8, std2_10, std2_12, std2_14 = [], [], [], [], [], []
diff2_4, diff2_6, diff2_8, diff2_10, diff2_12, diff2_14 = [], [], [], [], [], []

p3_4, p3_6, p3_8, p3_10, p3_12, p3_14 = [], [], [], [], [], []
mean3_4, mean3_6, mean3_8, mean3_10, mean3_12, mean3_14 = [], [], [], [], [], []
std3_4, std3_6, std3_8, std3_10, std3_12, std3_14 = [], [], [], [], [], []
diff3_4, diff3_6, diff3_8, diff3_10, diff3_12, diff3_14 = [], [], [], [], [], []

Ahom_4, Ahom_6, Ahom_8, Ahom_10, Ahom_12, Ahom_14 = [], [], [], [], [], []
t_homs_4, t_homs_6, t_homs_8, t_homs_10, t_homs_12, t_homs_14 = [], [], [], [], [], []
T_homs_4, T_homs_6, T_homs_8, T_homs_10, T_homs_12, T_homs_14 = [], [], [], [], [], []

u_4, u_6, u_8, u_10, u_12, u_14 = [], [], [], [], [], []
bet_4, bet_6, bet_8, bet_10, bet_12, bet_14 = [], [], [], [], [], []
area_4, area_6, area_8, area_10, area_12, area_14 = [], [], [], [], [], []
mus_4, mus_6, mus_8, mus_10, mus_12, mus_14 = [], [], [], [], [], []
cvs_4, cvs_6, cvs_8, cvs_10, cvs_12, cvs_14 = [], [], [], [], [], []
area_hom_4, area_hom_6, area_hom_8, area_hom_10, area_hom_12, area_hom_14 = [], [], [], [], [], []



# BIASES:

bias_w = 0.1
value = 1
percent = 50

# 4 TASK SWITCHES

T = 2200
rate = 0.002

# SIMULATION:

mus = [6.176923]
#mus = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
#devs = [4.0, 3.0, 2.0, 1.0]
cvs = [0.25, 0.50, 0.75, 1.0]


for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_4(mu, cv*mu, 1.0, times4)
        
#mu = 4.8   
#for cv in tqdm(cvs):
    #perf_4(mu, cv*mu, pe=1.0, tmin1 = 500, tmax1 = 800, tmin2 = 800, tmax2 = 1100, tmin3 = 1100, tmax3 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_4, 'cv': cv_4, 'performance1_hom': Ahom_4, 'performance1_het%': p_4, 'performance1_het_mean': mean_4,
        'performance1_het_dev': std_4, 'performance1_diff': diff_4, 'performance2_hom': t_homs_4, 'performance2_het%': p2_4,
        'performance2_het_mean': mean2_4, 'performance2_het_dev': std2_4, 'performance2_diff': diff2_4, 'performance3_hom': T_homs_4,               'performance3_het%': p3_4, 'performance3_het_mean': mean3_4, 'performance3_het_dev': std3_4, 'performance3_diff': diff3_4}
                                        
df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_4, 'cv': cvs_4, 'u_vector': u_4, 'perf het > hom?': bet_4, 'performance_hom': area_hom_4, 
         'performance1_het_dev': area_4}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_2_biases_distr.csv', index=False)


# 6 TASK SWITCHES

T = 2200
rate = 0.003

# SIMULATION:
mus = [4.758974]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_6(mu, cv*mu, 1.0, times6)

        
#mu = 2.9 
#for cv in tqdm(cvs):
    #perf_6(mu, cv*mu, pe=1.0, tmin1 = 400, tmax1 = 600, tmin2 = 600, tmax2 = 800, tmin3 = 800, tmax3 = 1000, tmin4 = 1000, tmax4 = 1200,              tmin5 = 1200, tmax5 = 1400)
    

# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_6, 'cv': cv_6, 'performance1_hom': Ahom_6, 'performance1_het%': p_6, 'performance1_het_mean': mean_6,
        'performance1_het_dev': std_6, 'performance1_diff': diff_6, 'performance2_hom': t_homs_6, 'performance2_het%': p2_6,
        'performance2_het_mean': mean2_6, 'performance2_het_dev': std2_6, 'performance2_diff': diff2_6, 'performance3_hom': T_homs_6,               'performance3_het%': p3_6, 'performance3_het_mean': mean3_6, 'performance3_het_dev': std3_6, 'performance3_diff': diff3_6}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_6, 'cv': cvs_6, 'u_vector': u_6, 'perf het > hom?': bet_6, 'performance_hom': area_hom_6, 
         'performance1_het_dev': area_6}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_2_biases_distr.csv', index=False)



# 8 TASK SWITCHES:

T = 2200
rate = 0.004

# SIMULATION:

mus = [3.543590]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_8(mu, cv*mu, 1.0, times8)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_8, 'cv': cv_8, 'performance1_hom': Ahom_8, 'performance1_het%': p_8, 'performance1_het_mean': mean_8,
        'performance1_het_dev': std_8, 'performance1_diff': diff_8, 'performance2_hom': t_homs_8, 'performance2_het%': p2_8,
        'performance2_het_mean': mean2_8, 'performance2_het_dev': std2_8, 'performance2_diff': diff2_8, 'performance3_hom': T_homs_8,               'performance3_het%': p3_8, 'performance3_het_mean': mean3_8, 'performance3_het_dev': std3_8, 'performance3_diff': diff3_8}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_8, 'cv': cvs_8, 'u_vector': u_8, 'perf het > hom?': bet_8, 'performance_hom': area_hom_8, 
         'performance1_het_dev': area_8}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_2_biases_distr.csv', index=False)


# 10 TASK SWITCHES:

T = 2200
rate = 0.005

# SIMULATION:
mus = [2.125641]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_10(mu, cv*mu, 1.0, times10)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_10, 'cv': cv_10, 'performance1_hom': Ahom_10, 'performance1_het%': p_10, 'performance1_het_mean': mean_10,
        'performance1_het_dev': std_10, 'performance1_diff': diff_10, 'performance2_hom': t_homs_10, 'performance2_het%': p2_10,
        'performance2_het_mean': mean2_10, 'performance2_het_dev': std2_10, 'performance2_diff': diff2_10, 'performance3_hom': T_homs_10,               'performance3_het%': p3_10, 'performance3_het_mean': mean3_10, 'performance3_het_dev': std3_10, 'performance3_diff': diff3_10}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_10, 'cv': cvs_10, 'u_vector': u_10, 'perf het > hom?': bet_10, 'performance_hom': area_hom_10, 
         'performance1_het_dev': area_10}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_2_biases_distr.csv', index=False)



# 12 TASK SWITCHES:

T = 2200
rate = 0.006

# SIMULATION:
mus = [0.910256]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_12(mu, cv*mu, 1.0, times12)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_12, 'cv': cv_12, 'performance1_hom': Ahom_12, 'performance1_het%': p_12, 'performance1_het_mean': mean_12,
        'performance1_het_dev': std_12, 'performance1_diff': diff_12, 'performance2_hom': t_homs_12, 'performance2_het%': p2_12,
        'performance2_het_mean': mean2_12, 'performance2_het_dev': std2_12, 'performance2_diff': diff2_12, 'performance3_hom': T_homs_12,               'performance3_het%': p3_12, 'performance3_het_mean': mean3_12, 'performance3_het_dev': std3_12, 'performance3_diff': diff3_12}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_12, 'cv': cvs_12, 'u_vector': u_12, 'perf het > hom?': bet_12, 'performance_hom': area_hom_12, 
         'performance1_het_dev': area_12}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_2_biases_distr.csv', index=False)

# 14 TASK SWITCHES:

T = 2200
rate = 0.007

# SIMULATION:
mus = [0.1]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_14(mu, cv*mu, 1.0, times14)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_14, 'cv': cv_14, 'performance1_hom': Ahom_14, 'performance1_het%': p_14, 'performance1_het_mean': mean_14,
        'performance1_het_dev': std_14, 'performance1_diff': diff_14, 'performance2_hom': t_homs_14, 'performance2_het%': p2_14,
        'performance2_het_mean': mean2_14, 'performance2_het_dev': std2_14, 'performance2_diff': diff2_14, 'performance3_hom': T_homs_14,               'performance3_het%': p3_14, 'performance3_het_mean': mean3_14, 'performance3_het_dev': std3_14, 'performance3_diff': diff3_14}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_2_biases.csv', index=False)

#Data2
data1 = {'mu': mus_14, 'cv': cvs_14, 'u_vector': u_14, 'perf het > hom?': bet_14, 'performance_hom': area_hom_14, 
         'performance1_het_dev': area_14}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_2_biases_distr.csv', index=False)

#--------------------------------------------------------------------------------------------------------------------------------------
#1 BIASES

# list to store data:

mu_4, mu_6, mu_8, mu_10, mu_12, mu_14 = [], [], [], [], [], []
cv_4, cv_6, cv_8, cv_10, cv_12, cv_14 = [], [], [], [], [], []
#dev_2, dev_4, dev_6, dev_8 = [], [], [], [], []

p_4, p_6, p_8, p_10, p_12, p_14 = [], [], [], [], [], []
mean_4, mean_6, mean_8, mean_10, mean_12, mean_14 = [], [], [], [], [], []
std_4, std_6, std_8, std_10, std_12, std_14 = [], [], [], [], [], []
diff_4, diff_6, diff_8, diff_10, diff_12, diff_14 = [], [], [], [], [], []

p2_4, p2_6, p2_8, p2_10, p2_12, p2_14 = [], [], [], [], [], []
mean2_4, mean2_6, mean2_8, mean2_10, mean2_12, mean2_14 = [], [], [], [], [], []
std2_4, std2_6, std2_8, std2_10, std2_12, std2_14 = [], [], [], [], [], []
diff2_4, diff2_6, diff2_8, diff2_10, diff2_12, diff2_14 = [], [], [], [], [], []

p3_4, p3_6, p3_8, p3_10, p3_12, p3_14 = [], [], [], [], [], []
mean3_4, mean3_6, mean3_8, mean3_10, mean3_12, mean3_14 = [], [], [], [], [], []
std3_4, std3_6, std3_8, std3_10, std3_12, std3_14 = [], [], [], [], [], []
diff3_4, diff3_6, diff3_8, diff3_10, diff3_12, diff3_14 = [], [], [], [], [], []

Ahom_4, Ahom_6, Ahom_8, Ahom_10, Ahom_12, Ahom_14 = [], [], [], [], [], []
t_homs_4, t_homs_6, t_homs_8, t_homs_10, t_homs_12, t_homs_14 = [], [], [], [], [], []
T_homs_4, T_homs_6, T_homs_8, T_homs_10, T_homs_12, T_homs_14 = [], [], [], [], [], []

u_4, u_6, u_8, u_10, u_12, u_14 = [], [], [], [], [], []
bet_4, bet_6, bet_8, bet_10, bet_12, bet_14 = [], [], [], [], [], []
area_4, area_6, area_8, area_10, area_12, area_14 = [], [], [], [], [], []
mus_4, mus_6, mus_8, mus_10, mus_12, mus_14 = [], [], [], [], [], []
cvs_4, cvs_6, cvs_8, cvs_10, cvs_12, cvs_14 = [], [], [], [], [], []
area_hom_4, area_hom_6, area_hom_8, area_hom_10, area_hom_12, area_hom_14 = [], [], [], [], [], []



# BIASES:

bias_w = 0.1
value = 1
percent = 25

# 4 TASK SWITCHES

T = 2200
rate = 0.002

# SIMULATION:

mus = [6.176923]
#mus = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
#devs = [4.0, 3.0, 2.0, 1.0]
cvs = [0.25, 0.50, 0.75, 1.0]


for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_4(mu, cv*mu, 1.0, times4)
        
#mu = 4.8   
#for cv in tqdm(cvs):
    #perf_4(mu, cv*mu, pe=1.0, tmin1 = 500, tmax1 = 800, tmin2 = 800, tmax2 = 1100, tmin3 = 1100, tmax3 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_4, 'cv': cv_4, 'performance1_hom': Ahom_4, 'performance1_het%': p_4, 'performance1_het_mean': mean_4,
        'performance1_het_dev': std_4, 'performance1_diff': diff_4, 'performance2_hom': t_homs_4, 'performance2_het%': p2_4,
        'performance2_het_mean': mean2_4, 'performance2_het_dev': std2_4, 'performance2_diff': diff2_4, 'performance3_hom': T_homs_4,               'performance3_het%': p3_4, 'performance3_het_mean': mean3_4, 'performance3_het_dev': std3_4, 'performance3_diff': diff3_4}
                                        
df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_4, 'cv': cvs_4, 'u_vector': u_4, 'perf het > hom?': bet_4, 'performance_hom': area_hom_4, 
         'performance1_het_dev': area_4}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./4_task_sw_1_biases_distr.csv', index=False)



# 6 TASK SWITCHES

T = 2200
rate = 0.003

# SIMULATION:
mus = [4.758974]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_6(mu, cv*mu, 1.0, times6)

        
#mu = 2.9 
#for cv in tqdm(cvs):
    #perf_6(mu, cv*mu, pe=1.0, tmin1 = 400, tmax1 = 600, tmin2 = 600, tmax2 = 800, tmin3 = 800, tmax3 = 1000, tmin4 = 1000, tmax4 = 1200,              tmin5 = 1200, tmax5 = 1400)
    

# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_6, 'cv': cv_6, 'performance1_hom': Ahom_6, 'performance1_het%': p_6, 'performance1_het_mean': mean_6,
        'performance1_het_dev': std_6, 'performance1_diff': diff_6, 'performance2_hom': t_homs_6, 'performance2_het%': p2_6,
        'performance2_het_mean': mean2_6, 'performance2_het_dev': std2_6, 'performance2_diff': diff2_6, 'performance3_hom': T_homs_6,               'performance3_het%': p3_6, 'performance3_het_mean': mean3_6, 'performance3_het_dev': std3_6, 'performance3_diff': diff3_6}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_6, 'cv': cvs_6, 'u_vector': u_6, 'perf het > hom?': bet_6, 'performance_hom': area_hom_6, 
         'performance1_het_dev': area_6}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./6_task_sw_1_biases_distr.csv', index=False)




# 8 TASK SWITCHES:

T = 2200
rate = 0.004

# SIMULATION:

mus = [3.543590]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_8(mu, cv*mu, 1.0, times8)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_8, 'cv': cv_8, 'performance1_hom': Ahom_8, 'performance1_het%': p_8, 'performance1_het_mean': mean_8,
        'performance1_het_dev': std_8, 'performance1_diff': diff_8, 'performance2_hom': t_homs_8, 'performance2_het%': p2_8,
        'performance2_het_mean': mean2_8, 'performance2_het_dev': std2_8, 'performance2_diff': diff2_8, 'performance3_hom': T_homs_8,               'performance3_het%': p3_8, 'performance3_het_mean': mean3_8, 'performance3_het_dev': std3_8, 'performance3_diff': diff3_8}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_8, 'cv': cvs_8, 'u_vector': u_8, 'perf het > hom?': bet_8, 'performance_hom': area_hom_8, 
         'performance1_het_dev': area_8}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./8_task_sw_1_biases_distr.csv', index=False)



# 10 TASK SWITCHES:

T = 2200
rate = 0.005

# SIMULATION:
mus = [2.125641]
for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_10(mu, cv*mu, 1.0, times10)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_10, 'cv': cv_10, 'performance1_hom': Ahom_10, 'performance1_het%': p_10, 'performance1_het_mean': mean_10,
        'performance1_het_dev': std_10, 'performance1_diff': diff_10, 'performance2_hom': t_homs_10, 'performance2_het%': p2_10,
        'performance2_het_mean': mean2_10, 'performance2_het_dev': std2_10, 'performance2_diff': diff2_10, 'performance3_hom': T_homs_10,               'performance3_het%': p3_10, 'performance3_het_mean': mean3_10, 'performance3_het_dev': std3_10, 'performance3_diff': diff3_10}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_10, 'cv': cvs_10, 'u_vector': u_10, 'perf het > hom?': bet_10, 'performance_hom': area_hom_10, 
         'performance1_het_dev': area_10}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./10_task_sw_1_biases_distr.csv', index=False)



# 12 TASK SWITCHES:

T = 2200
rate = 0.006

# SIMULATION:
mus = [0.910256]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_12(mu, cv*mu, 1.0, times12)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_12, 'cv': cv_12, 'performance1_hom': Ahom_12, 'performance1_het%': p_12, 'performance1_het_mean': mean_12,
        'performance1_het_dev': std_12, 'performance1_diff': diff_12, 'performance2_hom': t_homs_12, 'performance2_het%': p2_12,
        'performance2_het_mean': mean2_12, 'performance2_het_dev': std2_12, 'performance2_diff': diff2_12, 'performance3_hom': T_homs_12,               'performance3_het%': p3_12, 'performance3_het_mean': mean3_12, 'performance3_het_dev': std3_12, 'performance3_diff': diff3_12}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_12, 'cv': cvs_12, 'u_vector': u_12, 'perf het > hom?': bet_12, 'performance_hom': area_hom_12, 
         'performance1_het_dev': area_12}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./12_task_sw_1_biases_distr.csv', index=False)


# 14 TASK SWITCHES:

T = 2200
rate = 0.007

# SIMULATION:
mus = [0.1]

for mu in tqdm(mus):
    for cv in tqdm(cvs):
        perf_14(mu, cv*mu, 1.0, times14)
        
#mu = 0.2
#for cv in tqdm(cvs):
    #perf_8(mu, cv*mu, pe=1.0, tmin1 = 350, tmax1 = 500, tmin2 = 500, tmax2 = 650, tmin3 = 650, tmax3 = 800, tmin4 = 800, tmax4 = 950,                 tmin5 = 950, tmax5 = 1100, tmin6 = 1100, tmax6 = 1250, tmin7 = 1250, tmax7 = 1400)


# Creazione del dataframe utilizzando le liste di dati
data = {'mu': mu_14, 'cv': cv_14, 'performance1_hom': Ahom_14, 'performance1_het%': p_14, 'performance1_het_mean': mean_14,
        'performance1_het_dev': std_14, 'performance1_diff': diff_14, 'performance2_hom': t_homs_14, 'performance2_het%': p2_14,
        'performance2_het_mean': mean2_14, 'performance2_het_dev': std2_14, 'performance2_diff': diff2_14, 'performance3_hom': T_homs_14,               'performance3_het%': p3_14, 'performance3_het_mean': mean3_14, 'performance3_het_dev': std3_14, 'performance3_diff': diff3_14}

df = pd.DataFrame(data)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_1_biases.csv', index=False)

#Data2
data1 = {'mu': mus_14, 'cv': cvs_14, 'u_vector': u_14, 'perf het > hom?': bet_14, 'performance_hom': area_hom_14, 
         'performance1_het_dev': area_14}
                                        
df = pd.DataFrame(data1)

# Salvataggio del dataframe in un file CSV
df.to_csv('./14_task_sw_1_biases_distr.csv', index=False)


#-----------------------------------------------------------------------------------------------------------------------------------------

# Assicurati che tutte le liste abbiano la stessa lunghezza
max_length = max(len(times4), len(times6), len(times8), len(times10), len(times12), len(times14))

# Riempi le liste più corte con valori mancanti (e.g., np.nan)
times4 += [None] * (max_length - len(times4))
times6 += [None] * (max_length - len(times6))
times8 += [None] * (max_length - len(times8))
times10 += [None] * (max_length - len(times10))
times12 += [None] * (max_length - len(times12))
times14 += [None] * (max_length - len(times14))

# Crea il DataFrame
data = {'time_4': times4, 'time_6': times6, 'time_8': times8, 'time_10': times10, 'time_12': times12, 'time_14': times14}
df = pd.DataFrame(data)

# Salva il DataFrame in un file CSV
df.to_csv('./switching_times.csv', index=False)
