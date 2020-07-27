%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hospital level analysis
% Real data: U = 1, C = 41
% A: 41-by-41 infectivity matrix, 
% mu: 41-by-1 intensity vector
% g: 1-by-1 
% A: 1 by 1 by 41
% mu: 1 by 41
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

D = 41;
U = 1;
num_events = 1033; % after dropping duplicate rows

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading in medical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seqs_big = struct();
seqs_big_train = struct();
seqs_big_test = struct();

% all
time_vect = load('time_all.mat');
mark_vect = load('mark_all.mat');
seqs_big(1).Time = time_vect.time_all;
seqs_big(1).Mark = mark_vect.mark_all;
seqs_big(1).Start = 0;
seqs_big(1).Stop = 545; % maximum timestamp over all units
seqs_big(1).Feature = [];

% train
time_vect_train = load('time_all_train.mat');
mark_vect_train = load('mark_all_train.mat');
seqs_big_train(1).Time = time_vect_train.time_all;
seqs_big_train(1).Mark = mark_vect_train.mark_all;
seqs_big_train(1).Start = 0;
seqs_big_train(1).Stop = 545; % maximum timestamp over all units
seqs_big_train(1).Feature = [];

% test
time_vect_test = load('time_all_test.mat');
mark_vect_test = load('mark_all_test.mat');
seqs_big_test(1).Time = time_vect_test.time_all;
seqs_big_test(1).Mark = mark_vect_test.mark_all;
seqs_big_test(1).Start = 0;
seqs_big_test(1).Stop = 545; % maximum timestamp over all units
seqs_big_test(1).Feature = [];

% half = ceil((length(time_vect.time_all))*(1/2));
% seqs_big_train(1).Time = time_vect.time_all(1:half);
% seqs_big_test(1).Time = time_vect.time_all(half+1:end);
% seqs_big_train(1).Mark = mark_vect.mark_all(1:half);
% seqs_big_test(1).Mark = mark_vect.mark_all(half+1:end);
% seqs_big_train(1).Start = 0;
% seqs_big_test(1).Start = 0;
% seqs_big_train(1).Stop = 545; % maximum timestamp over all units
% seqs_big_test(1).Stop = 545; % maximum timestamp over all units
% seqs_big_train(1).Feature = [];
% seqs_big_test(1).Feature = [];

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

alg.alphaLR = 0.2; %0.2; %0.2; %0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.002; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

model.kernel = 'gauss';
model.w = 2;
model.landmark = 0; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.outer = 1;
alg.inner = 5;

nTest = 2; %5 % number of tests
nSeg = 3; %10 %number of chunks of data
nNum = floor((num_events/2)/nSeg);

A_multi = zeros(D,U,D); %[];
mu_multi = [];
LL = zeros(1, nSeg, nTest);

for n = 1:nTest
    for i = 1:nSeg
        model.A = rand(D,U,D)./(U*D^2); % when not using Initialization_Basis
        model.mu = rand(D,U)./D; % when not using Initialization_Basis
        
        part_seqs_big_train = struct();
        part_seqs_big_train.Time = seqs_big_train.Time(1:i*nNum);
        part_seqs_big_train.Mark = seqs_big_train.Mark(1:i*nNum);
        part_seqs_big_train.Start = 0;
        part_seqs_big_train.Stop = 545;
        
        %model = Initialization_Basis(part_seqs_big_train); % lowers LL (but L (units) is incorrect for MTmu)
        output = Learning_MLE_Basis(part_seqs_big_train, model, alg); %seqs_big
        %output = Learning_MLE_Basis_MTmu(part_seqs_big_train, model, alg);
        
        LL(1,i,n) = Loglike_Basis(seqs_big_test, output, alg); % seqs_big_test
        disp(LL(1,i,n));
    end
    %LL = Loglike_Basis(seqs_big, output, alg);
    %disp(LL);
    A_multi = A_multi + output.A; %vertcat(A_multi, output.A);
    disp(size(A_multi));
    %mu_multi(1,i,n) = vertcat(mu_multi, output.mu);
    disp("Done with a test run.");
end

% LL_mean = mean(LL, 3); 
% LL_std = std(LL, 0, 3);
% LL_stats = [LL_mean; LL_std];
LL_mean = mean(LL, 3); 
LL_std = std(LL, 0, 3);
LL_stats = [LL_mean; LL_std];

disp(LL_stats);

% A_multi_mean = mean(A_multi);
% A_multi_std = std(A_multi);
% A_multi_stats = [A_multi_mean; A_multi_std];

% mu_multi_mean = mean(mu_multi, 3); 
% mu_multi_std = std(mu_multi, 0, 3);
% mu_multi_stats = [mu_multi_mean; mu_multi_std];

A_multi_mean = A_multi/nTest;

writematrix(LL_stats, "data/MMHP_hospital_LL.csv"); 
writematrix(A_multi_mean, "data/MMHP_hospital_A.csv"); 
%writematrix(mu_multi_stats, "data/MMHP_hospital_mu.csv"); 

