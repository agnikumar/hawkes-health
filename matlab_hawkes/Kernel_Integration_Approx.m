function G = Kernel_Integration_Approx(dt, para)

G = zeros(length(dt(:)), size(para.g,2));

M = size(para.g,1);
Nums = ceil(dt./para.dt);
%Nums = ceil(dt./para.dt) + 1; % FOR CUSTOM SEQUENCES
for i = 1:length(dt(:))
    if Nums(i)<=M
        G(i,:) = sum(para.g(1:Nums(i),:)).*para.dt;
    else
        G(i,:) = sum(para.g).*para.dt;
    end
end
