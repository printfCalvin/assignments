load('0_1_2.mat')
%(a)
[U, S, ~] = svd(X);
Y_pca = U(:, 1:2)' * X;
%plotimages(reshape(X,8,8,300), Y_pca, 0.1, 1);

%(b)
X1 = X(:, 1:100);
X2 = X(:, 101:200);
X3 = X(:, 201:300);
Sw = cov(X1') + cov(X2') + cov(X3');
[d, n] = size(X);
St = cov(X');
Sb = St - Sw;
W = Sw \ Sb;
[V, D] = eig(W);
Y_fda = V(:, 1:2)' * X;
%plotimages(reshape(X,8,8,300), Y_fda, 0.03, 1);

%(c)
mu0 = mean(Y_pca(:, 1:100), 2);
mu1 = mean(Y_pca(:, 101:200), 2);
mu2 = mean(Y_pca(:, 201:300), 2);
sig0 = 1 / (100 - 2) * (Y_pca(:, 1:100) - mu0) * (Y_pca(:, 1:100) - mu0)';
sig1 = 1 / (100 - 2) * (Y_pca(:, 101:200) - mu1) * (Y_pca(:, 101:200) - mu1)';
sig2 = 1 / (100 - 2) * (Y_pca(:, 201:300) - mu2) * (Y_pca(:, 201:300) - mu2)';
sig = 1 / 3 * (sig0 + sig1 + sig2);
findboundary(mu0, mu1, sig)
findboundary(mu0, mu2, sig)
findboundary(mu1, mu2, sig)

Y_pca_q = [Y_pca; Y_pca(1, :).^2; Y_pca(2, :).^2];
mu0_q = mean(Y_pca_q(:, 1:100), 2);
mu1_q = mean(Y_pca_q(:, 101:200), 2);
mu2_q = mean(Y_pca_q(:, 201:300), 2);
sig0_q = 1 / (100 - 4) * (Y_pca_q(:, 1:100) - mu0_q) * (Y_pca_q(:, 1:100) - mu0_q)';
sig1_q = 1 / (100 - 4) * (Y_pca_q(:, 101:200) - mu1_q) * (Y_pca_q(:, 101:200) - mu1_q)';
sig2_q = 1 / (100 - 4) * (Y_pca_q(:, 201:300) - mu2_q) * (Y_pca_q(:, 201:300) - mu2_q)';
sig_q = 1 / 3 * (sig0_q + sig1_q + sig2_q);
findboundary(mu0_q, mu1_q, sig_q)
findboundary(mu0_q, mu2_q, sig_q)
findboundary(mu1_q, mu2_q, sig_q)

%d
plotimages(reshape(X,8,8,300), Y_pca, 0.1, 1);
hold on;
fimplicit(@(x, y) (mu1' - mu0') / sig * [x;y] + 1/2 * (mu0' / sig * mu0 - mu1' / sig * mu1));
fimplicit(@(x, y) (mu2' - mu0') / sig * [x;y] + 1/2 * (mu0' / sig * mu0 - mu2' / sig * mu2));
fimplicit(@(x, y) (mu1' - mu2') / sig * [x;y] + 1/2 * (mu2' / sig * mu2 - mu1' / sig * mu1));


plotimages(reshape(X,8,8,300), Y_pca, 0.1, 1);
hold on;
fimplicit(@(x, y) (mu1_q' - mu0_q') / sig_q * [x;y;x^2;y^2] + 1/2 * (mu0_q' / sig_q * mu0_q - mu1_q' / sig_q * mu1_q));
fimplicit(@(x, y) (mu2_q' - mu0_q') / sig_q * [x;y;x^2;y^2] + 1/2 * (mu0_q' / sig_q * mu0_q - mu2_q' / sig_q * mu2_q));
fimplicit(@(x, y) (mu1_q' - mu2_q') / sig_q * [x;y;x^2;y^2] + 1/2 * (mu2_q' / sig_q * mu2_q - mu1_q' / sig_q * mu1_q));

%e

error_rate(X)

%f


function plotimages(images, Y ,scale,proportion)
% function plotimages(images, Y, scale, proportion)
%
% images = images, must be in a 3-dimensional matrix (x by y by n)
% for example if X is 64 by 400 and size of each image is 8 by 8, images=reshape(X,8,8,400);
%
% Y = where to plot the image (Y(1,:) by Y(2,:))
%
% proportion = proportion of the data to be ploted (proportion <= 1).
% for example if there are 400 data points proportion = 1, plots
% all 400 data points and proportion = 0.5 plot only 200 data points 
% (i.e. 1th, 3th, 5th, ...)
% Ali Ghodsi 2006

%Y=normr(Y);
inc=floor(1/proportion);
% scale = scale of each image wrt to figure size (scale<1)
%scale=10;
xoff=0;
yoff=0;

xlim = get(gca, 'XLim');
ylim = get(gca, 'YLim');


width = (xlim(2) - xlim(1)) * scale;
height = (ylim(2) - ylim(1)) * scale;

colormap(gray);

image_width = size(images,1);
image_height = size(images,2);
n_images = size(images,3);

xs = Y(1,:) + xoff;
ys = Y(2,:) + yoff;
hold on
for counter = 1:inc:n_images
   
	current_image = 1-reshape(images(:,:,counter), [image_width image_height]);
	imagesc( ...
		[xs(counter)		xs(counter)+width], ...
		[ys(counter)+height	ys(counter)], ...
		current_image' ...
	);
end
xlabel ('x')
ylabel ('y')
hold off
end
function A = findboundary(mu0, mu1, sig)
    temp = mu1' / sig - mu0' / sig;
    [~, n] = size(temp);
    A = reshape([temp, 1/2 * (mu0' / sig * mu0 - mu1' / sig * mu1)], 1, n + 1);
end
function num = lda(X, mu)
    [~, n] = size(mu);
    min = (X - mu(:,1))' * (X - mu(:,1));
    num = 0;
    for i = 2:n
        temp = (X - mu(:,i))' * (X - mu(:,i));
        if min > temp
            min = temp;
            num = i - 1;
        end
    end
end
function e = error_rate(X)
    e = 0;
    mu0 = mean(X(:, 1:100), 2);
    mu1 = mean(X(:, 101:200), 2);
    mu2 = mean(X(:, 201:300), 2);
    mu = [mu0, mu1, mu2];
    for i = 1:300
        if 1 <= i && i <= 100 && lda(X(:, i), mu) ~= 0
            e = e + 1;
        elseif 101 <= i && i <= 200 && lda(X(:, i), mu) ~= 1
            e = e + 1;
        elseif 201 <= i && i <= 300 && lda(X(:, i), mu) ~= 2
            e = e + 1;
        end
    end
    e = e / 300;
end









