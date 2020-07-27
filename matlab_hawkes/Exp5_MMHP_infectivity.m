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

D = 20; %30;
U = 1;
num_events = 1033; % after dropping duplicate rows

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading in medical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% all
time_vect = load('time_all_lots.mat'); % top 30 units
mark_vect = load('mark_all_lots.mat');
seqs_big(1).Time = time_vect.time; % .time_all
seqs_big(1).Mark = mark_vect.mark; % .mark_all
seqs_big(1).Start = 0;
seqs_big(1).Stop = 545; % maximum timestamp over all units
seqs_big(1).Feature = [];

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
alg.alphaS = 10; %0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.002; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

model.kernel = 'gauss';
model.w = 0.5; %2; % 1
model.landmark = 100; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0; %0
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.outer = 4;
alg.inner = 20; %50;

nTest = 20; %1; %5 % number of tests

A_multi = zeros(D,U,D); %[];
mu_multi = [];

for n = 1:nTest
    model.A = rand(D,U,D)./(U*D^2); % when not using Initialization_Basis
    model.mu = rand(D,U)./D; % when not using Initialization_Basis

    output = Learning_MLE_Basis(seqs_big, model, alg); 
    %model = Initialization_Basis(part_seqs_big_train.file_struct); % lowers LL (but L (units) is incorrect for MTmu)
    %output = Learning_MLE_Basis_MTmu(part_seqs_big_train, model, alg);  
    
    A_multi = A_multi + output.A; %vertcat(A_multi, output.A);
    mu_multi = [mu_multi; transpose(output.mu)];
    disp("Done with a test run.");
end

A_multi_mean = A_multi/nTest;
mu_multi_stats = [mean(mu_multi); std(mu_multi)];

writematrix(A_multi_mean, "new_data/MMHP_hospital_A_plot_single_top20_match_3.csv"); 
writematrix(mu_multi_stats, "new_data/MMHP_hospital_mu_plot_single_top20_match_3.csv"); 

disp(size(mu_multi_stats));
