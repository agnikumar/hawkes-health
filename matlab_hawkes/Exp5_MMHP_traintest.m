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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading in medical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seqs_all = struct();
seqs_train = struct();
seqs_test = struct();
for n = 1:U % total number of units
    time_vect = load(sprintf('time_%d.mat', n));
    mark_vect = load(sprintf('mark_%d.mat', n));
    seqs_all(n).Time = time_vect.time;
    seqs_all(n).Mark = mark_vect.mark;
    seqs_all(n).Start = 0;
    seqs_all(n).Stop = 545; % maximum timestamp over all units
    seqs_all(n).Feature = [];
    
    half = ceil((length(time_vect.time))/2);
    seqs_train(n).Time = time_vect.time(1:half);
    seqs_test(n).Time = time_vect.time(half+1:end);
    seqs_train(n).Mark = mark_vect.mark(1:half);
    seqs_test(n).Mark = mark_vect.mark(half+1:end);
    seqs_train(n).Start = 0;
    seqs_test(n).Start = 0;
    seqs_train(n).Stop = 545; % maximum timestamp over all units
    seqs_test(n).Stop = 545; % maximum timestamp over all units
    seqs_train(n).Feature = [];
    seqs_test(n).Feature = [];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.LowRank = 0; %1;
alg.Sparse = 0;
alg.GroupSparse = 0;
alg.storeLL = 0;
alg.storeErr = 0;
alg.updatemu = 1; % for Learning_MLE_MTmu
alg.thres = 1e-5;
alg.rho = 0.1;
alg.Tmax = [];

alg.alphaLR = 10; %0.2; %0.2; %0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.002; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

model.kernel = 'gauss';
model.w = 1; %0.002; %2;
model.landmark = 40:10:440; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.outer = 1;
alg.inner = 5;

nTest = 5; % number of tests
nSeg = 10; % number of chunks of data
nNum = floor(U/nSeg);

A_multi = [];
mu_multi = [];
LL = zeros(1, nSeg, nTest);

for n = 1:nTest
    for i = 1:nSeg
        model.A = rand(D,U,D)./(U*D^2); % using Initialization_Basis
        model.mu = rand(D,U)./D; % using Initialization_Basis
        %model = Initialization_Basis(seqs_train(1:i*nNum)); % lowers LL (but L (units) is incorrect for MTmu)
        output = Learning_MLE_Basis(seqs_train(1:i*nNum), model, alg);
        %output = Learning_MLE_Basis_MTmu(seqs_train, model, alg);
        
%         A_multi = vertcat(A_multi, output.A);
%         mu_multi = vertcat(mu_multi, output.mu);
        LL(1,i,n) = Loglike_Basis(seqs_test, output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1:i*nNum), output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1), output, alg);
    end
    disp("Done with a test run.");
end

LL_mean = mean(LL, 3); 
LL_std = std(LL, 0, 3);
LL_stats = [LL_mean; LL_std];

disp(LL_stats);

% A_multi_mean = mean(A_multi);
% A_multi_std = std(A_multi);
% A_multi_stats = [A_multi_mean; A_multi_std];
% 
% mu_multi_mean = mean(mu_multi);
% mu_multi_std = std(mu_multi);
% mu_multi_stats = [mu_multi_mean; mu_multi_std];

%writematrix(LL_stats, "data/MMHP_LL_redo_none.csv"); 
%writematrix(A_multi_stats, "data/MMHP_A_redo.csv"); 
%writematrix(mu_multi_stats, "data/MMHP_mu_redo.csv"); 
