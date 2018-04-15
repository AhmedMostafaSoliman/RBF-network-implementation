clear;
data_train= dbmoon(1000,1,10,6);
data_test = dbmoon(1000,1,10,6);
num_samples = size(data_train,1);
k=10; 
initial_sigma =max(pdist(data_train(:,1:2),'euclidean'))/sqrt(2*num_samples);
centers_kmeans = kmeans(k,data_train(:,1:2));
[w,sigma,centers] = rbf_train(centers_kmeans,data_train(:,1:2),data_train(:,3),initial_sigma);
%Testing the trained network
net = zeros(num_samples,k);
in = zeros(num_samples,k); %inputs to the output node and activations of hidden neurons

for center_idx=1:k
    net(:,center_idx) = sum((data_test(:,1:2)-repmat(centers(center_idx,:),num_samples,1)).^2,2);
    in(:,center_idx)  = exp(-net(:,center_idx)./(2*sigma(center_idx).^2)) ;  
end
out = in * w';
sum_of_squared_error = sum((data_test(:,3)-out).^2)
out=round(out);
classification_accuracy = (1- nnz(out-data_test(:,3))/num_samples)*100