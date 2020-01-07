from functions import *
from classes import *
import numpy as np

def Simulation_Branch_HP(para, options):
    '''
    Simulate Hawkes processes as Branch processes
    '''

    #Seqs = struct('Time', [], ...
    #              'Mark', [], ...
    #              'Start', [], ...
    #              'Stop', [], ...
    #              'Feature', []);
    Seqs = []
    
    for n in range(options.N):
        
        # the 0-th generation, simulate exogeneous events via Poisson processes
        History = Simulation_Thinning_Poisson(para.mu, 0, options.Tmax)
        current_set = History
        
        for k in range(options.GenerationNum):
            future_set = []
            for i in range(current_set.shape[1]):
                ti = current_set[0,i] # check for correctness
                ui = current_set[1,i] # check for correctness
                t = 0
                
                phi_t = ImpactFunction(ui, t, para)
                mt = np.sum(phi_t)
                
                while t < options.Tmax - ti:

                    s = np.random.exponential(mt)
                    U = np.random.uniform(0, 1)

                    phi_ts = ImpactFunction(ui, t+s, para)
                    mts = np.sum(phi_ts)

                    print('s={}, v={})'.format(s, mts/mt))     
                    if t+s > options.Tmax-ti or U>mts/mt:
                        t = t+s
                    else:
                        u = np.random.uniform(0, 1) * mts
                        sumIs = 0
                        for d in range(len(phi_ts)):
                            sumIs = sumIs + phi_ts(d)
                            if sumIs >= u:
                                break
                        index = d

                        t = t+s
                        #future_set=[future_set,[t+ti;index(1)]]; # TODO: fix
                        future_set = np.concatenate((future_set, np.concatenate((t+ti, index[0]))), 1) # fixed

                    phi_t = ImpactFunction(ui, t, para)
                    mt = np.sum(phi_t)
            
            if future_set.size == 0 or History.shape[1] > options.Nmax:
                break
            else:
                current_set = future_set
                History = [History, current_set]

        
        Seqs_n = Sequence() # added
        _, index = np.sort(History[:]) # check
        Seqs_n.Time = History[0,index-1] # check
        Seqs_n.Mark = History[1,index-1] # check
        Seqs_n.Start = 0
        Seqs_n.Stop = options.Tmax
        index = np.nonzero(Seqs_n.Time <= options.Tmax)
        Seqs_n.Time = Seqs_n.Time(index)
        Seqs_n.Mark = Seqs_n.Mark(index)
        
        if n % 10 == 0 or n==options.N:
            print('seq={}/{}, event={}'.format(n, options.N, len(Seqs_n.Mark)))

        Seqs[n] = Seqs_n # added
    return Seqs