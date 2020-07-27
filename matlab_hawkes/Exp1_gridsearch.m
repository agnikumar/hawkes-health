
clear
     
% simulation options 
options.N = 20; % 200; % the number of sequences (10 for ps3 experiment)
options.Nmax = 100; %100; % the maximum number of events per sequence
options.Tmax = 50; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 10;
D = 1; %3; % the dimension of Hawkes processes (number of classes)
%nTest = 1;
%nSeg = 5;

% simulation parameters
% disp('Approximate simulation of Hawkes processes via branching process')
% disp('Complicated gaussian kernel')
para.kernel = 'gauss';
para.w = 1.5; 
para.landmark = 0:1:9; %0:4:12;
L = length(para.landmark); % should be the same as number of units (?)
para.mu = rand(D,1)/D;
para.A = zeros(D, D, L);
for l = 1:L
    para.A(:,:,l) = (0.5^l)*(0.5+rand(D));
end
para.A = 0.9*para.A./max(abs(eig(sum(para.A,3))));
para.A = reshape(para.A, [D, L, D]);
Seqs = Simulation_Branch_HP(para, options);

alg.LowRank = 1; % on
alg.Sparse = 1; % on

alg.rho = 0.1;
alg.outer = 10; % number of runs
alg.inner = 5; % number of loops (as related to convergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 1; % on
alg.storeLL = 1; % on 

model.kernel = 'gauss';
model.w = 2;
model.landmark = para.landmark;
alg.inner_g = 100;
model.D = L; %options.N; 
model.dt = 0.02;
% for error calculations
alg.truth.A = para.A; % added
alg.truth.mu = para.mu; % added

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M_vals = {10, 100, 1000}; 
% alpha_vals = {10, 100, 1000};
% alphaLR_vals = {0, 0.25, 0.5};
% alphaS_vals = {0, 0.0025, 0.005};
% alphaLR_mu_vals = {0, 0.0025, 0.005};

M_vals = {10, 100, 500, 1000}; 
alpha_vals = {10, 100, 1000, 10000};
alphaLR_vals = {0, 0.2, 0.4, 0.6};
alphaS_vals = {0, 0.002, 0.004, 0.006};
alphaLR_mu_vals = {0, 0.002, 0.004, 0.006};

data = [];

for M_index = 1:length(M_vals)
        model.M = M_vals{M_index};
        
    for alpha_index = 1:length(alpha_vals)
        alg.alpha = alpha_vals{alpha_index};

        for alphaLR_index = 1:length(alphaLR_vals)
            alg.alphaLR = alphaLR_vals{alphaLR_index};

            for alphaS_index = 1:length(alphaS_vals)
                alg.alphaS = alphaS_vals{alphaS_index};

                for alphaLR_mu_index = 1:length(alphaLR_mu_vals)
                    alg.alphaLR_mu = alphaLR_mu_vals{alphaLR_mu_index};
                    
                    % run model
                    %output = model;
                    % initialize (Basis)
                    model.A = rand(D,L,D)./(L*D^2);
                    model.mu = rand(D,1)./D;
                    
                    % initialize (ODE)
                    model.g = rand(model.M, model.D); % global triggering kernel of length M
                    model.g = model.g./repmat(sum(model.g),[model.M,1]);
                    % for g error calculations 
                    model.truth = model; % added
                    model.truth.g = model.g; % added

                    output = Exp1_Basis_ODE_variant(Seqs, model, alg);
                    %disp("----------");
                    data = vertcat(data, output.params_LL_err);
                    
                end
            end
        end
    end
    disp("Finished a parameter group at the highest level.")
end

writematrix(data, "/Users/agnikumar/Documents/MEng_new/data/grid_4ps.csv");
disp("Wrote data to file.")
