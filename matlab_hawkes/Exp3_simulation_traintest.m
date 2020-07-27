%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trying method on simulation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real data: U = 41, C = 1
% Expect 41 (A, mu) pairs, and one global triggering kernel g
% A: 1 by 1 by 41
% mu: 1 by 41
% Looking at how LL changes are more data is included in training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

D = 1;
U = 41;

A_uni_vals = [];
mu_uni_vals = [];
LL_test_vals = [];

% seqs_all = struct();
% for n = 1:U % total number of units
%     time_vect = load(sprintf('time_%d.mat', n));
%     mark_vect = load(sprintf('mark_%d.mat', n));
%     seqs_all(n).Time = time_vect.time;
%     seqs_all(n).Mark = mark_vect.mark;
%     seqs_all(n).Start = 0;
%     seqs_all(n).Stop = 545; % maximum timestamp over all units
%     seqs_all(n).Feature = [];
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alg.LowRank = 0;
alg.Sparse = 0;
alg.GroupSparse = 0;
alg.alphaLR = 0.2; %0.2; %0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.002; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 1; % perviously, 4
alg.rho = 0.1;
alg.inner = 5; % perviously, 15
alg.thres = 1e-5;
alg.Tmax = [];
alg.storeErr = 0;
alg.storeLL = 0;

alg.updatemu = 1;

model.kernel = 'gauss';
model.w = 2; %0.002; %2;
model.landmark = 0; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0;
%model.kernel = 'exp';
%model.w = 1;
%model.landmark = 0;

nTest = 1; %5
nSeg = 5; % number of chunks of data
%nNum = U/nSeg;
nNum = floor((U*0.5)/nSeg);
%nNum = floor(U/nSeg);

%A_multi = zeros(1,nSeg,nTest);
%mu_multi = zeros(1,nSeg,nTest);
A_multi = [];
mu_multi = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation Sequence 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
para1.kernel = 'exp';
para1.w = 1.5; 
para1.landmark = zeros(1,41); %0:1:40; %0:4:12;
L = length(para1.landmark);
para1.mu = rand(D,1)/D;
para1.A = zeros(D, D, U);
for l = 1:L
    para1.A(:,:,l) = (0.5^l)*(0.5+rand(D));
end
para1.A = 0.9*para1.A./max(abs(eig(sum(para1.A,3))));
para1.A = reshape(para1.A, [D, U, D]);
options.N = 41; % the number of sequences
options.Nmax = 84; % the maximum number of events per sequence
options.Tmax = 545; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 10;
seqs_all = Simulation_Branch_HP(para1, options);

LL = zeros(1,nSeg,nTest);
for n = 1:nTest
    for i = 1:nSeg
        model.A = rand(D,U,D)./(U*D^2);
        model.mu = rand(D,U)./D;
        output = Learning_MLE_Basis_MTmu(seqs_all(1:floor(i*nNum)), model, alg);
        %A_multi(1,i,n) = output.A; % learned A
        %mu_multi(1,i,n) = output.mu; % learned mu
        A_multi = vertcat(A_multi, output.A);
        mu_multi = vertcat(mu_multi, output.mu);
        
        LL(1,i,n) = Loglike_Basis(seqs_all(floor(U*0.5)+1:U), output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1:i*nNum), output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1), output, alg);
    end
    disp("Done with a test run.");
end

LL_mean = mean(LL, 3); 
LL_std = std(LL, 0, 3);
LL_stats = [LL_mean; LL_std];

%A_multi_mean = mean(A_uni, 3); 
%A_multi_std = std(A_uni, 0, 3);
A_multi_mean = mean(A_multi);
A_multi_std = std(A_multi);
A_multi_stats = [A_multi_mean; A_multi_std];

%mu_multi_mean = mean(mu_multi, 3); 
%mu_multi_std = std(mu_multi, 0, 3);
mu_multi_mean = mean(mu_multi);
mu_multi_std = std(mu_multi);
mu_multi_stats = [mu_multi_mean; mu_multi_std];

%writematrix(LL_stats, "data/multi_LL_trial_3.csv"); 
%writematrix(A_multi_stats, "data/multi_A_trial_3.csv"); 
%writematrix(mu_multi_stats, "data/multi_mu_trial_3.csv"); 
