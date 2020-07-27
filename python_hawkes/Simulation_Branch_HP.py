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
    
    #Seqs = []
    Seqs = [None] * options.N
    
    for n in range(options.N):
        
        # the 0-th generation, simulate exogeneous events via Poisson processes
        History = Simulation_Thinning_Poisson(para.mu, 0, options.Tmax) # should be 2 x __
        current_set = History

        #print("History:", History)
        #print(History.shape)
        #print("-------------")
        
        for k in range(options.GenerationNum):
            #future_set = []
            future_set = np.empty((2,0))
            for i in range(current_set.shape[1]):
                ti = current_set[0,i] # check for correctness
                ui = current_set[1,i] # check for correctness
                t = 0
                
                #print("ui:", ui)
                #print("PARA:", para)

                phi_t = ImpactFunction(ui, t, para)
                mt = np.sum(phi_t)
                
                while t < options.Tmax - ti:

                    s = np.random.exponential(1/mt) #check!!
                    U = np.random.uniform(0, 1)

                    phi_ts = ImpactFunction(ui, t+s, para)
                    #print("phi_ts", phi_ts) # added
                    #print(phi_ts.shape) # is 3x4, but should be 3x1
                    mts = np.sum(phi_ts)

                    #print('s={}, v={}'.format(s, mts/mt))
                    #print((t+s > options.Tmax-ti) or (U>mts/mt)) # added     
                    if (t+s > options.Tmax-ti) or (U>mts/mt):
                        t = t+s
                    else:
                        #print("GOT INSIDE ELSE") # added
                        u = np.random.uniform(0, 1) * mts
                        sumIs = 0
                        #print("sumIs before", sumIs)
                        for d in range(len(phi_ts)):
                            sumIs = sumIs + phi_ts[d]
                            #print("sumIs", sumIs) # added
                            #print("phi_ts[d]", phi_ts[d]) # added
                            #print("u", u) # added
                            if sumIs >= u:
                                break
                        index = d

                        t = t+s
                        #future_set=[future_set,[t+ti;index(1)]]; # TODO: fix
                        #print("index", index) # added
                        #future_set = np.concatenate((future_set, np.concatenate((t+ti, index[0]))), 1) # fixed
                        t_ti_index = np.vstack((t+ti, index))
                        #print("t_ti_index", t_ti_index.shape) # added

                        future_set = np.hstack((future_set, t_ti_index))
                        #print("future_set shape", future_set.shape) # added

                    phi_t = ImpactFunction(ui, t, para)
                    mt = np.sum(phi_t)
                #print("Got outside while!!") # added
            if future_set.size == 0 or History.shape[1] > options.Nmax:
                break
            else:
                current_set = future_set
                #History = [History, current_set]
                History = np.hstack((History, current_set))

        
        Seqs_n = Sequence() # added
        _, index = np.sort(History[:]) # check
        #print("index", index) # added
        #print(index.shape) # should be 1x__, added
        index = index.astype(int) # added

        Seqs_n.Time = History[0,index-1] # check
        Seqs_n.Mark = History[1,index-1] # check
        Seqs_n.Start = 0
        Seqs_n.Stop = options.Tmax
        index = np.nonzero(Seqs_n.Time <= options.Tmax)
        Seqs_n.Time = Seqs_n.Time[index]
        Seqs_n.Mark = Seqs_n.Mark[index]
        
        if n % 10 == 0 or n==options.N:
            print('seq={}/{}, event={}'.format(n, options.N, len(Seqs_n.Mark)))
            #print(len(Seqs_n.Time))

        Seqs[n] = Seqs_n # added
    return Seqs