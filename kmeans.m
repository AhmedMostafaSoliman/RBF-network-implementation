function centers = kmeans(k,data)
%this function returns the cluster centers given the number of clusters and the data points

num_iterations = 10000;
num_samples = size(data,1);
centers = data(randsample(num_samples,k),:);  % choose k random points from the data to be the initial centers
distances = zeros(num_samples,k);
 
for it=1:num_iterations
    %step 1 assign each point to the nearest center
    for center_idx= 1:k
        distances(:,center_idx) = sqrt(sum((data-repmat(centers(center_idx,:),num_samples,1)).^2,2));    
    end
    [~ ,min_indices]= min(distances,[],2);
    %step 2 compute the new means 
    for center_idx = 1:k
        centers(center_idx,:) = mean(data(min_indices == center_idx,:));
    end
end

scatter(data(:,1),data(:,2),'b')
hold on
scatter(centers(:,1),centers(:,2),'r')
    

