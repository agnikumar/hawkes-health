%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real data: U = 41, C = 1
% Expect 41 (A, mu) pairs, and one global triggering kernel g
% A: 1 by 1 by 41
% mu: 1 by 41
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
UB = 1; %upper bound, also test others

seqs_all = struct();
for n = 1:41 % total number of units
    time_vect = load(sprintf('time_%d.mat', n));
    mark_vect = load(sprintf('mark_%d.mat', n));
    seqs_all(n).Time = time_vect.time;
    seqs_all(n).Mark = mark_vect.mark;
    seqs_all(n).Start = 0;
    seqs_all(n).Stop = 545; % maximum timestamp over all units
    seqs_all(n).Feature = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alg.LowRank = 1; % on
alg.Sparse = 1; % on
alg.alphaLR = 0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 5; % number of runs
alg.rho = 0.1;
alg.inner = 10; % number of loops (as related to convergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 0; % on, no error (real data)
alg.storeLL = 1; % on 

% for error calculations
%alg.truth.A = para.A; % added
%alg.truth.mu = para.mu; % added

% initialize (Basis)
D = 1; % one event type (mark), positive diagnosis
L = 41; % number of units
model.A = rand(D,L,D)./(L*D^2);
model.mu = rand(D,1)./D;
model.kernel = 'gauss';
model.w = 2;
%model.landmark = para.landmark;
model.landmark = 0:1:40; %0:4:12;

% initialize (ODE)
alg.inner_g = 100;
alg.alpha = 100; %10000;
model.M = 1000; %1000;
model.D = L; %options.N; 
model.dt = 0.02;
model.g = rand(model.M, model.D); % global triggering kernel of length M
model.g = model.g./repmat(sum(model.g),[model.M,1]);

% for error calculations 
%model.truth = model; % added
%model.truth.g = model.g; % added

model_real_all = model;
model_real_all = Exp4_Basis_ODE_custom(seqs_all, model_real_all, alg);
[A_both, Phi_both] = ImpactFunc_ODE(model_real_all);

%disp("Model errors (A, mu, g):");
%disp(size(model_real_all.err));
%disp(model_real_all.err); % order of error columns: A, mu, g
disp("Model log-likelihood (not NLL):"); % single column
disp(size(model_real_all.LL));
disp(model_real_all.LL);

