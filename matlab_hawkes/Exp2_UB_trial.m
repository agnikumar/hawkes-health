%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robustness analysis
% Setting upper bounds for A and mu to be 1, 5, 10, 20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
UB = 5; %upper bound, also test others
     
% simulation options 
options.N = 200; % 200; % the number of sequences
options.Nmax = 100; %100; % the maximum number of events per sequence
options.Tmax = 50; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 10;
D = 3; %1; % the dimension of Hawkes processes (number of classes)
%nTest = 1;
%nSeg = 5;

% simulation parameters
% disp('Approximate simulation of Hawkes processes via branching process')
% disp('Complicated gaussian kernel')
para.kernel = 'gauss';
para.w = 1.5; 
para.landmark = 0:4:12; %0:1:19;
L = length(para.landmark); % should be the same as number of units (?)

%para.mu = rand(D,1)/D;
para.mu = unifrnd(0, UB, [D, 1])/D; % with UB

para.A = zeros(D, D, L);
for l = 1:L
    %para.A(:,:,l) = (0.5^l)*(0.5+rand(D));
    para.A(:,:,l) = (0.5^l)*(0.5 + unifrnd(0, UB, [D, D])); % with UB
end
para.A = para.A./max(abs(eig(sum(para.A,3)))); % multiply by 0.9?
para.A = reshape(para.A, [D, L, D]);
Seqs1 = Simulation_Branch_HP(para, options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alg.LowRank = 1; % on
alg.Sparse = 1; % on
alg.alphaLR = 0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 1; % number of runs
alg.rho = 0.1;
alg.inner = 5; % number of loops (as related to convergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 1; % on
alg.storeLL = 1; % on 

% for error calculations
alg.truth.A = para.A; % added
alg.truth.mu = para.mu; % added

% initialize (Basis)
model.A = rand(D,L,D)./(L*D^2);
model.mu = rand(D,1)./D;
model.kernel = 'gauss';
model.w = 2;
model.landmark = para.landmark;

% initialize (ODE)
alg.inner_g = 100;
alg.alpha = 100; %10000;
model.M = 1000; %1000;
model.D = L; %options.N; 
model.dt = 0.02;
model.g = rand(model.M, model.D); % global triggering kernel of length M
model.g = model.g./repmat(sum(model.g),[model.M,1]);

% for error calculations 
model.truth = model; % added
model.truth.g = model.g; % added

model_both = model;
%model_both = Exp1_Basis_ODE(Seqs, model_both, alg);
model_both = Exp1_Basis_ODE_variant(Seqs1, model_both, alg);
%model_both = Exp1_Learning_MLE_Basis(Seqs, model_both, alg);
[A_both, Phi_both] = ImpactFunc_ODE(model_both);

disp("Model errors (A, mu, g):");
disp(size(model_both.err));
disp(model_both.err); % order of error columns: A, mu, g
disp("Model log-likelihood (not NLL):"); % single column
disp(size(model_both.LL));
disp(model_both.LL);

