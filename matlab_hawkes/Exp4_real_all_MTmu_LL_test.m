%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real data: U = 41, C = 1
% Expect 41 (A, mu) pairs, and one global triggering kernel g
% A: 1 by 1 by 41
% mu: 1 by 41
% Looking at how LL changes are more data is included in training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
UB = 1; %upper bound, also test others
U = 41; % number of units

seqs_all = struct();
for n = 1:U % total number of units
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
alg.alphaLR = 0.2; %0.2; %0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 1; % number of runs
alg.rho = 0.1;
alg.inner = 8; % number of loops (as related to convergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 0; % on, no error (real data)
alg.storeLL = 1; % on 

% initialize (ODE)
alg.inner_g = 100;
alg.alpha = 100000; %1000000; %100; %10000;

L = U;
D = 1;
alg.updatemu = 1;

model_MTmu.A = rand(D,L,D)./(L*D^2);
model_MTmu.mu = rand(D,L)./D;
model_MTmu.landmark = 0:1:40;
model_MTmu.kernel = 'gauss';
model_MTmu.w = 1; %0.002; %2;

LL_test_vals = [];
LL_vals = []; % LL over time
nSeg = 8; % number of chunks of data
%nNum = U/nSeg;
nNum = floor((U*0.5)/nSeg);
for i = 1:nSeg
    model_MTmu.A = rand(D,L,D)./(L*D^2);
    model_MTmu.mu = rand(D,L)./D;
    model_real_all_MTmu = Learning_MLE_Basis_MTmu(seqs_all(1:floor(i*nNum)), model_MTmu, alg);

    LL_test = Loglike_Basis(seqs_all(floor(U*0.5)+1:U), model_real_all_MTmu, alg);
    disp(LL_test);
    disp("---------------");
    LL_test_vals = vertcat(LL_test_vals, LL_test);
end

disp("Model log-likelihood (not NLL):"); % single column
%disp(size(model_real_all.LL));
%disp(model_real_all.LL);

%disp(LL_vals);
disp(LL_test_vals);

