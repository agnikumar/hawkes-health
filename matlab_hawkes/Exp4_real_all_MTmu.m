L = 41;
D = 1;
alg.updatemu = 1;
model_MTmu.A = rand(D,L,D)./(L*D^2);
model_MTmu.mu = rand(D,L)./D;

model_MTmu.landmark = 0:1:40;
model_MTmu.kernel = 'gauss';
model_MTmu.w = 2;

output_MTmu = Learning_MLE_Basis_MTmu(seqs_all, model_MTmu, alg);
disp("Done!");