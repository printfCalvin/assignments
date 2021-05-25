load('Ion.trin.mat');
load('Ion.test.mat');

test_err = zeros(1, 50);
train_err = zeros(1, 50);
weightPenalty = 0;
nodes = 4;

% for nodes = 1:50
%    err = nn(Xtrain, ytrain, Xtest, ytest, nodes, weightPenalty);
%    test_err(nodes) = err(1);
%    train_err(nodes) = err(2);
% end
%     
% plot(train_err)
% hold on
% plot(test_err)
nn(Xtrain, ytrain, Xtest, ytest, nodes, weightPenalty)

function err = nn(Xtrain, ytrain, Xtest, ytest, nodes, weightPenalty)
test_err = zeros(1, 100);
train_err = zeros(1, 100);

%initialize
m = size(Xtrain, 1); 			        %features
n = size(Xtrain, 2); 			        %datapoints
%nodes = 10; 			%number of hidden nodes
u1=rand(m,nodes)-0.5; 	%input-to-hidden weights
u2=rand(nodes,1)-0.5; 	%hidden-to-output weights
ro = 0.1; 			%learning rate

   
Z_output=zeros(n,1);
Z=zeros(nodes,n);
Z_output_test=zeros(n - 1,1);
Z_test=zeros(nodes,n - 1);

% TRAIN DATA
for epoch=1:100
    for i=1:n
        % Forward Pass
        %determine inputs to hidden layer
        A=u1'*Xtrain(:,i);
           
        %apply activation function to hidden layer weighted inputs
        for j=1:nodes
            Z(j,i)=1/(1+exp(-A(j)));
        end
               
        %apply weights to get fitted outputs;
        Z_output(i,1) = u2'*Z(:,i);
           
        % Backward Pass
        %output delta
        delta_O = -2*(ytrain(i)-Z_output(i,1));
    
        %tweak the hidden-output weights
        for j=1:nodes
            u2(j)=u2(j)-ro*(delta_O*Z(j)+2*weightPenalty*u2(j));
        end
           
        for j=1:nodes
            sigmaPrime=exp(-A(j))/(1+exp(-A(j)))^2;
            delta_H = sigmaPrime*delta_O*u2(j);
            u1(:,j)=u1(:,j)-ro*(delta_H*Xtrain(:,i)+2*weightPenalty*u1(:,j));
        end
        if i == n
            continue
        end
        A_test=u1'*Xtest(:,i);
        for j=1:nodes
            Z_test(j,i)=1/(1+exp(-A_test(j)));
        end
        Z_output_test(i,1) = u2'*Z_test(:,i);
    end
    yhat(:,1) = Z_output(:,1) > 0.5;
    train_err(epoch) = (yhat - ytrain)' * (yhat - ytrain);
    
    temp = Z_output_test(:,1) > 0.5;
    test_err(epoch) = (temp - ytest)' * (temp - ytest);
end
    %err = [train_err(100) test_err(100)];
    y00 = 0;
    y01 = 0;
    y10 = 0;
    y11 = 0;
    for i = 1:size(yhat)
        if ytrain(i) == 0 & yhat(i) == 0
            y00 = y00 + 1;
        elseif ytrain(i) == 1 & yhat(i) == 0
            y10 = y10 + 1;
        elseif ytrain(i) == 0 & yhat(i) == 1
            y01 = y01 + 1;
        else
            y11 = y11 + 1;
        end
    end
        err = [y00 y01 y10 y11];
end
