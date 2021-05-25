% load('linear_new.mat');
% get_ans(X, y, Xtest, ytest, 0, 1)
load('noisylinear_new_1.mat');
get_ans(X, y, Xtest, ytest, -3, 3)
% load('quadratic_new.mat');
% get_ans(X, y, Xtest, ytest, 0, 1)

function [b, b_0] = HardMarg(X, y)
    H = X .* y';
    H = H' * H;
    [~, n] = size(X);
    alphas = quadprog(H, -ones(n ,1), [], [], y', 0, zeros(n, 1), []);
    b = sum(alphas .* y .* X', 1)';
    idx = find(alphas > 0.0001);
    i = idx(1,:);
    b_0 = (1 - y(i,:) * b' * X(:, i)) / y(i,:);
end

function [b, b_0] = SoftMarg(X, y, gamma)
    H = X .* y';
    H = H' * H;
    [~, n] = size(X);
    b = zeros(n , 1);
    gamma_vec = zeros(n , 1) + gamma;
    alphas = quadprog(H, -ones(n ,1), [], [], y', 0, b, gamma_vec);
    b = sum(alphas .* y .* X', 1)';
    idx = find(alphas > 0.0001);
    i = idx(1,:);
    b_0 = (1 - y(i,:) * b' * X(:, i)) / y(i,:);
end

function [yhat] = classify(Xtest, b, b_0)
    y = Xtest' * b + b_0;
    yhat(y <= 0) = -1;
    yhat(y > 0) = 1;
end

function mme = cal_mme(y_test, y_hat)
    [n, ~] = size(y_test);
    count = 0;
    for i = 1:n
        if y_test(i,:) ~= y_hat(i, :)
            count = count + 1;
        end
    end
    mme = count / n;
end

function get_ans(X, y, Xtest, ytest, from, to)
    [bh, b_0h] = HardMarg(X, y);
    [bs, b_0s] = SoftMarg(X, y, 0.5);
    cls_1 = X(:, y == -1);
    cls_2 = X(:, y == 1);
    plot(cls_1(1,:), cls_1(2,:), '.')
    hold on
    plot(cls_2(1,:), cls_2(2,:), '.')
    hold on
    f = @(x) - bh(1,:) / bh(2,:) * x - b_0h / bh(2,:);
    line1 = fplot(f, [from , to]);
    hold on 
    f = @(x) - bs(1,:) / bs(2,:) * x - b_0s / bs(2,:);
    line2 = fplot(f, [from , to]);
    legend([line1 line2],'hard margin SVM','soft margin SVM');
    yhat = classify(Xtest, bh, b_0h);
    mme_hard = cal_mme(ytest, yhat')
    yhat = classify(Xtest, bs, b_0s);
    mme_soft = cal_mme(ytest, yhat')
end