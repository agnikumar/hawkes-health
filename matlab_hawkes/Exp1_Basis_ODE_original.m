function model = Exp1_Basis_ODE_original(Seqs, model, alg)
                                                        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Adding lambda_3 (low-rank regularization on mu) to Learning_MLE_Basis
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial 
Aest = model.A;        
muest = model.mu;

%GK = struct('intG', []);

if alg.LowRank
    UL = zeros(size(Aest));
    ZL = Aest;
    UL_mu = zeros(size(muest)); % added
    ZL_mu = muest; % added
end

if alg.Sparse
    US = zeros(size(Aest));
    ZS = Aest;
end

if alg.GroupSparse
    UG = zeros(size(Aest));
    ZG = Aest;
end

D = size(Aest, 1);

if alg.storeLL
    model.LL = zeros(alg.outer,1);
end
if alg.storeErr
    model.err = zeros(alg.outer, 3);
end

tic;
for o = 1:alg.outer
    
    DM = zeros(size(model.g)); % added
    CM = DM; % added
    
    rho = alg.rho * (1.1^o);
    
    for n = 1:alg.inner
        
        NLL = 0; % negative log-likelihood
        
        Amu = zeros(D, 1);
        Bmu = Amu;
        
        CmatA = zeros(size(Aest));
        AmatA = CmatA;
        BmatA = CmatA;
        
        if alg.LowRank
            %BmatA = BmatA + rho*(UL-ZL);
            BmatA = BmatA + rho*(UL-ZL) + rho*(UL_mu-ZL_mu); % added
            AmatA = AmatA + rho;
        end
        if alg.Sparse
            BmatA = BmatA + rho*(US-ZS);
            AmatA = AmatA + rho;
        end
        if alg.GroupSparse
            BmatA = BmatA + rho*(UG-ZG);
            AmatA = AmatA + rho;
        end
        
        % E-step: evaluate the responsibility using the current parameters    
        for c = 1:length(Seqs)
            if ~isempty(Seqs(c).Time)
                Time = Seqs(c).Time;
                Event = Seqs(c).Mark;
                Tstart = Seqs(c).Start;

                if isempty(alg.Tmax)
                    Tstop = Seqs(c).Stop;
                else
                    Tstop = alg.Tmax;
                    indt = Time < alg.Tmax;
                    Time = Time(indt);
                    Event = Event(indt);
                end

                
                Amu = Amu + Tstop - Tstart;

                dT = Tstop - Time;
                GK = Kernel_Integration(dT, model);
                %GK = Kernel_Integration_Approx(dT, model); % added

                Nc = length(Time);

                for i = 1:Nc

                    ui = Event(i);

                    BmatA(ui,:,:) = BmatA(ui,:,:)+...
                        double(Aest(ui,:,:)>0).*repmat( GK(i,:), [1,1,D] );

                    ti = Time(i);  
                    
                    % added -----------------------------------------
                    if n == alg.inner
                        Nums = min([ceil(dT(i)/model.dt), size(CM,1)]);
                        CM(1:Nums,:) = CM(1:Nums,:) + ...
                            repmat(sum(Aest(ui,:,:), 3), [Nums,1]);
                    end
                    % -----------------------------------------------

                    lambdai = muest(ui);
                    pii = muest(ui);
                    pij = [];

                    if i>1

                        tj = Time(1:i-1);
                        uj = Event(1:i-1);
                        
                        dt = ti - tj;
                        
                        gij = Kernel(dt, model);
                        %gij = Kernel_Approx(dt, model); % added
                            
                        auiuj = Aest(uj, :, ui);
                        pij = auiuj .* gij;
                        lambdai = lambdai + sum(pij(:));
                    end

                    NLL = NLL - log(lambdai);
                    pii = pii./lambdai;

                    if i>1
                        pij = pij./lambdai;
                        if ~isempty(pij) && sum(pij(:))>0
                            for j = 1:length(uj)
                                uuj = uj(j);
                                CmatA(uuj,:,ui) = CmatA(uuj,:,ui) - pij(j,:); 
                                
                                % added -----------------------------------
                                if n == alg.inner
                                    Nums = min([ceil(dt(j)/model.dt), size(DM,1)]);
                                    DM(Nums,:) = DM(Nums,:)+pij(j,:);
                                end
                                % -----------------------------------------
                            end
                        end
                    end

                    Bmu(ui) = Bmu(ui) + pii;

                end

                NLL = NLL + (Tstop-Tstart).*sum(muest);
                NLL = NLL + sum( sum( GK.*sum(Aest(Event,:,:),3) ) );
                %NLL = NLL + sum( sum( GK(c).intG.*sum(Aest(Event,:,:),3) ) );

            
            else
                warning('Sequence %d is empty!', c)
            end
        end
                
        % M-step: update parameters
        mu = Bmu./Amu;        
        if alg.Sparse==0 && alg.GroupSparse==0 && alg.LowRank==0
            A = -CmatA./BmatA;%( -BA+sqrt(BA.^2-4*AA*CA) )./(2*AA);
            A(isnan(A))=0;
            A(isinf(A))=0;
        else            
            A = ( -BmatA + sqrt(BmatA.^2 - 4*AmatA.*CmatA) )./(2*AmatA);
            A(isnan(A))=0;
            A(isinf(A))=0;
        end
        
        % check convergence
        Err=sum(sum(sum(abs(A-Aest))))/sum(abs(Aest(:)));
        Aest = A;
        muest = mu;
        model.A = Aest;
        model.mu = muest;
        fprintf('Outer=%d, Inner=%d, Obj=%f, RelErr=%f, Time=%0.2fsec\n',...
                o, n, NLL, Err, toc);
            
        if Err<alg.thres || (o==alg.outer && n==alg.inner)
            break;
        end    
    end
    
    if alg.LowRank
        threshold = alg.alphaLR/rho;
        ZL = SoftThreshold_LR( Aest+UL, threshold );
        UL = UL + (Aest-ZL);
        threshold_mu = alg.alphaLR_mu/rho; % added
        ZL_mu = SoftThreshold_LR(Aest+UL_mu, threshold_mu); % added
        UL_mu = UL_mu + (Aest-ZL_mu); % added
    end
    
    if alg.Sparse
        threshold = alg.alphaS/rho;
        ZS = SoftThreshold_S( Aest+US, threshold );
        US = US + (Aest-ZS);
    end

    if alg.GroupSparse
        threshold = alg.alphaGS/rho;
        ZG = SoftThreshold_GS( Aest+UG, threshold );
        UG = UG + (Aest-ZG);
    end   
    
    % added -----------------------------------------------
    DM = DM./model.dt;
    CM = CM./model.g;
    
    gest = model.g;

    for n = 1:alg.inner_g
        for m = 1:size(model.g, 1)
            switch m
                case 1
                    a = 2*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha*model.g(m+1,:);
                    c = -DM(m,:) * (model.dt^2);
                case size(model.g, 1)
                    a = 4*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha*model.g(m-1,:);
                    c = -DM(m,:) * (model.dt^2);
                otherwise
                    a = 4*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha * (model.g(m-1,:)+model.g(m+1,:));
                    c = -DM(m,:) * (model.dt^2);
            end
            
            gest(m,:) = (-b+sqrt(b.^2 - 4.*a.*c))./(2*a);
        end

        model.g = abs(gest);  
        model.g = model.g./repmat(sum(model.g),[model.M,1]);
    end
    % ----------------------------------------------------
    
    % added (log-likelihood calculation)
    if alg.storeLL
        Loglike = Loglike_HP_ODE(Seqs, model, alg);
        model.LL(o) = Loglike;
    end
    
    % added (A, mu, g error calculation)
    if alg.storeErr
        Err = zeros(1,3);
        Err(1) = norm(model.A(:) - alg.truth.A(:))/norm(alg.truth.A(:));
        Err(2) = norm(model.mu(:) - alg.truth.mu(:))/norm(alg.truth.mu(:));
        model_ODE_eval = model.truth;
        model_ODE_eval.g = (model.g - model.truth.g).^2;
        model_eval = model.truth;
        model_eval.g = (model.truth.g).^2;
        Err(3) = (Kernel_Integration_Approx(model.dt, model_ODE_eval)) / ...
                 (Kernel_Integration_Approx(model.dt, model_eval));
        model.err(o,:) = Err;
    end
    
end
