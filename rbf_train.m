function [w,sigma,centers] = rbf_train(centers,train_in,train_target,initial_sigma)
eita_w=1;
eita_sigma=1;
eita_mu=1;
k= size(centers,1);
sigma = 6/k*initial_sigma*ones(k,1)'; %make the sigma adaptive to the number of clusters
w = 2*rand(k,1)'-1;
num_samples = size(train_in,1);
net = zeros(num_samples,k);
in = zeros(num_samples,k); %inputs to the output node and activations of hidden neurons

for i = 1:30000
    %forward pass
    for center_idx=1:k
        net(:,center_idx) = sum((train_in-repmat(centers(center_idx,:),num_samples,1)).^2,2);
        in(:,center_idx)  = exp(-net(:,center_idx)./(2*sigma(center_idx).^2)) ;   
    end
    out = in * w';
    %backward pass
    dEdOut = (out-train_target)';
    dW = dEdOut*in; % weight update rule
    dSigma = (dEdOut*(net .* in)).*w./(sigma.^3); %gaussian std deviation update rule
    dx = zeros(num_samples,k);dy = zeros(num_samples,k);
    for center_idx= 1:k 
        dx(:,center_idx) =  train_in(:,1) - centers(center_idx,1); 
        dy(:,center_idx) =  train_in(:,1) - centers(center_idx,2);
    end
    dCentersX = (dEdOut *(dx .* in).*w./(2*sigma.^2))';
    dCentersY = (dEdOut *(dy .* in).*w./(2*sigma.^2))';
    w = w - eita_w/num_samples *dW;
    sigma = sigma - eita_sigma/num_samples*dSigma;
    centers(:,1) = centers(:,1) - eita_mu/num_samples *dCentersX;
    centers(:,2) = centers(:,2) - eita_mu/num_samples *dCentersY;
end



