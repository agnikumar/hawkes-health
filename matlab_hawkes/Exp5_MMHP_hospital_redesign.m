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
time_filenames = {'time_train_1.mat', 'time_train_2.mat', 'time_train_3.mat', ...
                  'time_train_4.mat', 'time_train_5.mat', 'time_train_6.mat', ...
                  'time_train_7.mat', 'time_train_8.mat', 'time_train_9.mat', ...
                  'time_train_10.mat'};
mark_filenames = {'mark_train_1.mat', 'mark_train_2.mat', 'mark_train_3.mat', ...
                  'mark_train_4.mat', 'mark_train_5.mat', 'mark_train_6.mat', ...
                  'mark_train_7.mat', 'mark_train_8.mat', 'mark_train_9.mat', ...
                  'mark_train_10.mat'};
struct_filenames = {'train_1_struct.mat', 'train_2_struct.mat', 'train_3_struct.mat', ...
                  'train_4_struct.mat', 'train_5_struct.mat', 'train_6_struct.mat', ...
                  'train_7_struct.mat', 'train_8_struct.mat', 'train_9_struct.mat', ...
                  'train_10_struct.mat'};

% training subset
for pos = 1:10
    time_file = time_filenames{pos};
    mark_file = mark_filenames{pos};
    struct_file = struct_filenames{pos};
    file_struct = convert_to_struct(time_file, mark_file);
    save(struct_file, 'file_struct');
end

% testing set
seqs_big_test = convert_to_struct('time_test.mat', 'mark_test.mat'); % struct

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.LowRank = 0; %1;
alg.Sparse = 1;
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
model.w = 2;
model.landmark = 0; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.outer = 1;
alg.inner = 5;

nTest = 5; %5 % number of tests
nSeg = 10; %10 %number of chunks of data
nNum = floor((num_events/2)/nSeg);

A_multi = zeros(D,U,D); %[];
mu_multi = [];
LL = zeros(1, nSeg, nTest);

for n = 1:nTest
    for i = 1:nSeg
        model.A = rand(D,U,D)./(U*D^2); % when not using Initialization_Basis
        model.mu = rand(D,U)./D; % when not using Initialization_Basis
        
        part_seqs_big_train = load(sprintf('train_%d_struct.mat', i));
        
        %model = Initialization_Basis(part_seqs_big_train.file_struct); % lowers LL (but L (units) is incorrect for MTmu)
        output = Learning_MLE_Basis(part_seqs_big_train.file_struct, model, alg); %seqs_big
        %output = Learning_MLE_Basis_MTmu(part_seqs_big_train, model, alg);
        
        LL(1,i,n) = Loglike_Basis(seqs_big_test, output, alg); % seqs_big_test
        %disp(LL(1,i,n));
        
    end
    A_multi = A_multi + output.A; %vertcat(A_multi, output.A);
    %disp(size(A_multi));
    
    mu_multi = [mu_multi; transpose(output.mu)];
    
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

A_multi_mean = A_multi/nTest;
mu_multi_stats = [mean(mu_multi.'); std(mu_multi.')];

writematrix(LL_stats, "data/MMHP_hospital_LL_sparse_2.csv"); 
writematrix(A_multi_mean, "data/MMHP_hospital_A_sparse_2.csv"); 
writematrix(mu_multi_stats, "data/MMHP_hospital_mu_sparse_2.csv"); 
