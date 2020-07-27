# Simulating simple cases with mixed (Basis, ODE) code

from Learning_MLE_MIXED import *
import pandas as pd

#Seqs = struct('Time', {[3, 4.5, 6, 7, 10], [2, 4, 6, 8, 10]}, 'Mark', [1, 1, 1, 1, 1], 'Start', 0, 'Stop', 10);

#Seqs = Sequence()
#time_list = 
#Seqs_n.Time = np.reshape(time_list, (1, time_list.shape[0]))

# Creating small simulation sequence
df = pd.DataFrame(data={'time': [3, 4.5, 6, 7, 10, 2, 4, 6, 8, 10], 'unit': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']})
df_num_units = df['unit'].nunique()
df_sorted = df.sort_values(by=['unit', 'time'])
df_grouped = df_sorted.groupby(['unit'])
df_grouped_list = list(df_grouped)

df_Seqs = [None] * df_num_units
for n in range(df_num_units):
	Seqs_n = Sequence()
	time_list = np.asarray(df_grouped_list[n][1]['time'])
	Seqs_n.Time = np.reshape(time_list, (1, time_list.shape[0]))
	#Seqs_n.Mark = np.ones((1, Seqs_n.Time.shape[1]), dtype=int) # because D = 1
	Seqs_n.Mark = np.zeros((1, Seqs_n.Time.shape[1]), dtype=int) # because D = 1
	Seqs_n.Start = 0 #np.min(Seqs_n.Time)
	Seqs_n.Stop = 10 #np.max(Seqs_n.Time)
	df_Seqs[n] = Seqs_n

df_unit_names = []
for tup in df_grouped_list:
	df_unit_names.append(tup[0])

# Learning parameters
classes = 1 # only Cdiff-positive events
units = 2 # two distinct units (sequences)

# For learning kernel function g

model1 = Model()
model1.M = 3
model1.dt = 0.02
model1.g = [0.3, 0.1, 0.6, 0.8, 0.2, 0.9]
model1.g = np.reshape(model1.g, (model1.M, units), order='F')

model1.A = [2.1, 4.3]
model1.A = np.reshape(model1.A, (classes, units, classes))
model1.mu = [0.06] # make 1x2 eventually, fix mu dimensions in code
model1.mu = np.reshape(model1.mu, (classes, 1)) # fix mu dimensions in code

model1.LL = 1 # true
model1.err = 1 # true


alg1 = Algorithm()
alg1.LowRank = 0 # false, make true
alg1.Sparse = 0 # false
alg1.GroupSparse = 0 # false
alg1.outer = 1 # 5
alg1.rho = 0.1
alg1.inner = 1 # 8
alg1.thres = 1e-5
alg1.Tmax = np.empty((0,0)) # consider removing later
alg1.storeErr = 1
alg1.storeLL = 1

alg1.inner_g = 100
alg1.alpha = 10000

alg1.truth_mu = [0.06] # same as model.mu
alg1.truth_A = [2.1, 4.3] # same as model.A
alg1.alphaLR = 0.5
alg1.alphaS = 0.5 # currently off
alg1.alphaGS = 0.5 # currently off

model_MIXED = Learning_MLE_MIXED(df_Seqs, model1, alg1)
learned_A, learned_mu, learned_g = model_MIXED

print("Outputted Alpha (all):", learned_A)
print("Outputted Alpha shape (all):", learned_A.shape)

print("Outputted mu (all):", learned_mu)
print("Outputted mu shape (all):", learned_mu.shape)

print("Outputted g (all):", learned_g)
print("Outputted g shape (all):", learned_g.shape)

#print("Outputted NLL (all):", NLL)


