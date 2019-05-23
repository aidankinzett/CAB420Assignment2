%% a)
close all
clear

% load the iris data restricted to the first two features
load('iris.txt');
iris = iris(:,1:2);

% plot the data to see the clustering
scatter(iris(:,1), iris(:,2), 15, 'b*');

%% b) run k-means on the data

%% K=5
K1=5;
initial_k5 = [
    4.68    3.22;
    5.48    3.95;
    4.52    2.32;
    6.18    3.06;
    7.2     3.2;
];

centroids = initial_k5;
for i = 1:100
    idx = findClosestCentroids(iris, centroids);
    centroids = computeCentroids(iris, idx, K1);
end

% plot
figure; hold on;
plotDataPoints(iris, idx, K1);
plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15);
title('K-means clustering k=5');

%% K=20
K2=20;
initial_k20 = [  
    4.41	  3.23;
    5.37	  3.24;
    5.69	  2.22;
    5.69	  3.08;
    4.84	  2.86;
    5.83	  2.91;
    5.28	  2.36;
    6.84	  2.70;
    5.47	  4.02;
    5.21	  3.17;
    5.94	  2.33;
    4.97	  2.08;
    6.09	  2.83;
    4.47	  3.41;
    7.42	  3.45;
    5.06	  3.73;
    7.07	  3.50;
    4.94	  3.55;
    7.03	  2.87;
    7.62	  2.90;
];

centroids = initial_k20;
for i = 1:10
    idx = findClosestCentroids(iris, centroids);
    centroids = computeCentroids(iris, idx, K2);
end

% plot
figure; hold on;
plotDataPoints(iris, idx, K2);
plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15);
title('K-means clustering k=20');

%% c) agglomerative clustering
sLink = linkage(iris, 'single');
cLink = linkage(iris, 'complete');

%% 5 clusters
colors5 = jet(5);
clust = cluster(sLink, 'maxclust', 5);
figure;
scatter(iris(:,1), iris(:,2), 15, colors5(clust,:), 'filled');
title('Single linkage agglomerative clustering with 5 clusters');

clust = cluster(cLink, 'maxclust', 5);
figure;
scatter(iris(:,1), iris(:,2), 15, colors5(clust,:), 'filled');
title('Complete linkage agglomerative clustering with 5 clusters');


%% 20 clusters

colors20 = jet(20);
clust = cluster(sLink, 'maxclust', 20);
figure;
scatter(iris(:,1), iris(:,2), 15, colors20(clust,:), 'filled');
title('Single linkage agglomerative clustering wtih 20 clusters');

clust = cluster(cLink, 'maxclust', 20);
figure;
scatter(iris(:,1), iris(:,2), 15, colors20(clust,:), 'filled');
title('Complete linkage agglomerative clustering with 20 clusters');

%% d) EM Gaussian

clear;
% Load data
load('iris.txt');
iris = [iris(:,1), iris(:,2)];

% set the colormaps for the different number of clusters
colors5 = jet(5);
colors20 = jet(20);

%% 5 components
% TODO: change the inital clusters
K = 5;
initial_clusters = [
    5.0    3.5;
    4.5    2;
    5.7    2.7;
    6.5    3.0;
    5.2    3.9;
];

% Perform EM on Gaussian mixture model
[assign, clusters, ~, ~] = emCluster(iris, 5, initial_clusters);

% Plot the results

figure; hold on;
scatter(iris(:,1), iris(:,2), 15, colors5(assign,:), 'filled');
for i = 1:K
    plotGauss2D(clusters.mu(i,:), clusters.Sig(:,:,i), 'k', 'linewidth', 1);
end
title('EM Gausian mixture Model with 5 Components');

%% 20 components
% TODO: change the initial clusters
K = 20;
initial_clusters = [
    6.4913    2.9333;    
    4.4814    2.9917;    
    6.2679    2.9460;    
    6.8588    3.0600;
    4.7738    3.2812;    
    4.9284    2.4325;    
    5.7432    3.0969;    
    7.7151    2.8911;
    5.8724    2.7003;    
    6.4796    2.8164;    
    6.3604    2.5873;    
    5.4780    3.4583;
    7.9528    3.8910;    
    6.4738    3.2295;    
    6.4996    3.2526;    
    6.1055    2.8347;
    4.3266    3.0099;    
    4.9465    3.1763;    
    5.1536    3.3008;    
    4.8142    3.4620;
];

% Perform EM on Gaussian mixture model
[assign, clusters, ~, ~] = emCluster(iris, 20, initial_clusters);

% Plot the results
figure; hold on;
scatter(iris(:,1), iris(:,2), 15, colors20(assign,:), 'filled');
for i = 1:K
    plotGauss2D(clusters.mu(i,:), clusters.Sig(:,:,i), 'k', 'linewidth', 1);
end
title('EM Gausian mixture Model with 20 Components');

