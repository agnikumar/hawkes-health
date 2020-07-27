# Putting data into correct format for MMHP processing 

from classes_skeleton import *
#from Learning_MLE_ODE_medical import *
from Learning_MLE_ODE_medical_clean import *

import numpy as np
import pandas as pd 

D = 1 # dimension of Hawkes process (only one type of event, C. diff positive)

# --------------------------------------------------------------------
# Creating Seqs: df_one (df_one_Seqs)
# --------------------------------------------------------------------
df_one = pd.read_csv('data/event_times_ond_d_1314_ds0_one.csv')

df_one_num_units = df_one['unit'].nunique()
df_one_sorted = df_one.sort_values(by=['unit', 'time'])
df_one_grouped = df_one_sorted.groupby(['unit'])
df_one_grouped_list = list(df_one_grouped)

df_one_Seqs = [None] * df_one_num_units
for n in range(df_one_num_units):
	Seqs_n = Sequence()
	time_list = np.asarray(df_one_grouped_list[n][1]['time'])
	Seqs_n.Time = np.reshape(time_list, (1, time_list.shape[0]))
	#Seqs_n.Mark = np.ones((1, Seqs_n.Time.shape[1]), dtype=int) # because D = 1
	Seqs_n.Mark = np.zeros((1, Seqs_n.Time.shape[1]), dtype=int) # because D = 1
	Seqs_n.Start = np.min(Seqs_n.Time)
	Seqs_n.Stop = np.max(Seqs_n.Time)
	df_one_Seqs[n] = Seqs_n

df_one_unit_names = []
for tup in df_one_grouped_list:
	df_one_unit_names.append(tup[0])

'''
# Testing
print(df_one_Seqs[0].Time)
print(df_one_Seqs[0].Time.shape)
print("-------------------")
print(df_one_Seqs[0].Mark)
print(df_one_Seqs[0].Mark.shape)
print("-------------------")
print(df_one_Seqs[0].Start)
print("-------------------")
print(df_one_Seqs[0].Stop)
print(len(df_one_Seqs)) # should be 41
'''

# --------------------------------------------------------------------
# Creating Seqs: df_all (df_all_Seqs)
# --------------------------------------------------------------------
df_all = pd.read_csv('data/event_times_ond_d_1314_ds0_all.csv')

df_all_num_units = df_all['unit'].nunique()
df_all_sorted = df_all.sort_values(by=['unit', 'time'])
df_all_grouped = df_all_sorted.groupby(['unit'])
df_all_grouped_list = list(df_all_grouped)

df_all_Seqs = [None] * df_all_num_units
for n in range(df_all_num_units):
	Seqs_n = Sequence()
	time_list = np.asarray(df_all_grouped_list[n][1]['time'])
	Seqs_n.Time = np.reshape(time_list, (1, time_list.shape[0]))
	Seqs_n.Mark = np.zeros((1, Seqs_n.Time.shape[1]), dtype=int) # because D = 1
	Seqs_n.Start = np.min(Seqs_n.Time)
	Seqs_n.Stop = np.max(Seqs_n.Time)
	df_all_Seqs[n] = Seqs_n

df_all_unit_names = []
for tup in df_all_grouped_list:
	df_all_unit_names.append(tup[0])

'''
# Testing
print(df_all_Seqs[0].Time)
print(df_all_Seqs[0].Time.shape)
print("-------------------")
print(df_all_Seqs[0].Mark)
print(df_all_Seqs[0].Mark.shape)
print("-------------------")
print(df_all_Seqs[0].Start)
print("-------------------")
print(df_all_Seqs[0].Stop)
print(len(df_all_Seqs)) # should be 41
'''

# --------------------------------------------------------------------
# Instantiating parameters for Learning_MLE_ODE_medical.py
# Alpha: D x model.D x D = 1 x 41 x 1 (paper: D^2-by-units = 1x41)
# mu: D x 1 = 1 x 1 (paper:  D-by-U = 1x41) --> TODO: resolve difference
# g: M x model.D = M x 41
# --------------------------------------------------------------------
# Setting model
model = Model()
model.M = 1000 # number of samples of g
model.D = df_one_num_units # or, equivalently, df_all_num_units 
model.dt = 0.02 # sampling interval
model.g = np.matlib.rand(model.M, model.D)
model.g = model.g / np.tile(np.sum(model.g), (model.M, 1))
model.A = np.random.rand(D, model.D, D) / (model.D * D**2)
model.mu = np.matlib.rand(D, 1) / D

# Setting alg
alg = Algorithm()
alg.alpha = 10000
alg.inner = 3
alg.inner_g = 100
alg.outer = 8
#alg.thres = 1e-5
alg.Tmax = np.empty((0,0))

# --------------------------------------------------------------------
# Running for df_one
# --------------------------------------------------------------------
df_one_output = Learning_MLE_ODE_medical(df_one_Seqs, model, alg)
Alpha_one, mu_one, g_one, NLL_one = df_one_output 
print("Outputted Alpha (one):", Alpha_one)
print("Outputted Alpha shape (one):", Alpha_one.shape)

print("Outputted mu (one):", mu_one)
print("Outputted mu shape (one):", mu_one.shape)

print("Outputted g (one):", g_one)
print("Outputted g shape (one):", g_one.shape)

print("Outputted NLL (one):", NLL_one)
#print(df_one_output)

# --------------------------------------------------------------------
# Running for df_all
# --------------------------------------------------------------------
df_all_output = Learning_MLE_ODE_medical(df_all_Seqs, model, alg)
Alpha_all, mu_all, g_all, NLL_all = df_all_output 
print("Outputted Alpha (all):", Alpha_all)
print("Outputted Alpha shape (all):", Alpha_all.shape)

print("Outputted mu (all):", mu_all)
print("Outputted mu shape (all):", mu_all.shape)

print("Outputted g (all):", g_all)
print("Outputted g shape (all):", g_all.shape)

print("Outputted NLL (all):", NLL_all)
#print(df_one_output)

