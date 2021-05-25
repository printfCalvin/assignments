load('faces.mat');
training_data=[train_faces' train_nonfaces'];
test_data=[test_faces' test_nonfaces'];
n = size(training_data, 1);
Beta = rand(n, 1) - 0.5;
Y = [ones(size(train_faces, 1), 1); zeros(size(train_nonfaces, 1), 1)];
P = exp(Beta' * training_data) ./ (1 + exp(Beta' * training_data));
W = diag(P .* (1 - P));

for i = 1:100
    temp = training_data * W * training_data';
    if det(temp) ~= 0
        Beta = Beta + temp \ training_data * (Y - P);
    else
        Beta = Beta + pinv(temp) * training_data * (Y - P);
    end
    
end
Beta(:, 1:5)