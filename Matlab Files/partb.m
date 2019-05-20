%% clear and load stuff
clear
close all
X = load('data/faces.txt'); % load face dataset


%% understand the data
i=2;
img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24 image patch
imagesc(img); axis square; colormap gray; % display an image patch

%% a)
[m, n] = size(X);

mu = mean(X);
X0 = bsxfun(@minus, X, mu);
sigma = std(X0);
X0 = bsxfun(@rdivide, X0, sigma);

[U, S, V] = svd(X0);

W=U*S;

%% b)
ks = 1:10;
mserrs = zeros(size(ks));
for i=1:length(ks)
    X0_hat = W(:, 1:ks(i))*V(:, 1:ks(i))';
    mserrs(i) = mean(mean((X0-X0_hat).^2));
end

figure();
hold on;
plot(mserrs);
title('Mean Squared Error Vs K');
xlabel('K');
ylabel('MSE');
hold off;

%% c)
positive_pcs = {};
negative_pcs = {};
for j=1:10
    alpha = 2*median(abs(W(:, j)));
    positive_pcs{j} = mu + alpha*(V(:, j)');
    negative_pcs{j} = mu - alpha*(V(:, j)');
end

for i=1:3
    img = reshape(positive_pcs{i}, [24, 24]);
    figure('name', sprintf('Principal Direction (Positive) %d', i));
    imagesc(img);
    title(sprintf('Principal Direction (Positive) %d', i));
    axis square;
    colormap gray;

    img = reshape(negative_pcs{i}, [24, 24]);
    figure('name', sprintf('Principal Direction (Negative) %d', i));
    imagesc(img);
    title(sprintf('Principal Direction (Negative) %d', i))
    axis square;
    colormap gray;
end

%% d
idx = 80:100; % TODO: get random numbers working
figure('name', 'Latent Space Visualisation'); hold on; axis ij; colormap(gray);
title('Latent Space Visualisation')
xlabel('Principal Component 1');
ylabel('Principal Component 2');
range = max(W(idx, 1:2)) - min(W(idx, 1:2));
scale = [200 200]./range;

for i=idx
    imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:), 24, 24)); 
end

%% e
ks = [5, 10, 50];
faces = [354, 86, 129];

for f=1:length(faces)% for every face
    figure('name', sprintf('face %d', faces(f)));
    imagesc(reshape(X(faces(f),:), [24, 24]));
    axis square;
    colormap gray;
    title(sprintf('face %d', faces(f)));
    for i=1:length(ks) % for every k
        figure('name', sprintf('face %d reconstructed with %d pcs', faces(f), ks(i)));
        imagesc(reshape(W(faces(f), 1:ks(i))*V(1:576, 1:ks(i))', 24, 24));
        axis square;
        colormap gray;
        title(sprintf('face %d reconstructed with %d pcs', faces(f), ks(i)));
    end
end