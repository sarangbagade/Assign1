
% MLFFNN with single hidden layer for regression on data set 1
clear;

load('UnivariateData.mat');
load('prob4Weights.mat');
dimensionX = 1;
nodes = 25;
indexes = randperm(100,100);

trainX = x(indexes(1:70),1);
trainX(:,2) = 1;
validationX = x(indexes(70:90),1);
validationX(:,2) = 1;
testX = x(indexes(90:100),1);
testX(:,2) = 1;

trainY = fWithE(indexes(1:70),1);
validationY = fWithE(indexes(70:90),1);
testY = fWithE(indexes(90:100),1);

% % random initialization of weights
% W_h1 = rand(nodes,dimensionX+1); % 1 extra for bias parameter
% W_h2 = rand(nodes,1);
% 
% % finding the activation value of hidden nodes
% 
% a_h1(nodes,1) = zeros;
% s_h1(nodes,1) = zeros;
% %s_h1(nodes+1,1) = 1; % bias
% output(1,1) = zeros;
% 
% maxEpochs = 400;
% neta = 0.05; % learning parameter
% error = zeros;
% output(size(trainX,1),1) = zeros;
% error(maxEpochs,1) = zeros;
%     delta_h2(nodes,1) = zeros;
% 
% for i=1:1:maxEpochs
%     temp3 = zeros;
%     delta_h1 = zeros;
%     for j=1:1:size(trainX,1)
%         
%         %a_h1 = W_h1 * trainX(j,:)';
%         for k=1:1:nodes
%             s_h1(k,1) = 1 / (1 + exp(-1 * (W_h1(k,1) * trainX(j,1) + W_h1(k,2))));
%         end %s_h1(k,1) = 1 / (1 + exp(-1 .* a_h1(k,1)));
%         
%         output = W_h2' * s_h1;
%         
%         delta_ko = (trainY(j,1) - output);
%         
%         delta_h2(:,1) = delta_h2(:,1) + (neta * delta_ko) .* s_h1(:,1);
%         
%         %W_h2 = W_h2 + delta_h2;
%         
%         temp1(1:nodes,1) = zeros;
%         for k=1:1:nodes
%             temp1 = s_h1(k,1) * (1 - s_h1(k,1));
%             temp3 = temp3 + neta * delta_ko * W_h2(k,1) * temp1 * x(j,1);
% %             W_h1(k,1) = W_h1(k,1) + delta_h1;
%             delta_h1 = delta_h1 + neta * delta_ko * W_h2(k,1) * temp1;
% %             W_h1(k,2) = W_h1(k,1) + delta_h1;
%         end
%     end
%     
%     W_h2 = W_h2 + delta_h2 / size(trainX,1);
%     W_h1(k,1) = W_h1(k,1) + temp3 / size(trainX,1);
%     W_h1(k,2) = W_h1(k,1) + delta_h1 / size(trainX,1);
%     
%     for j=1:1:size(trainX,1)
%        
% %         a_h1 = W_h1 * trainX(j,:)';
%         for k=1:1:nodes
%             s_h1(k,1) = 1 / (1 + exp(-1 * (W_h1(k,1) *trainX(j,1) + W_h1(k,2))));
%         end %s_h1(k,1) = 1 / (1 + exp(-1 .* a_h1(k,1)));
%         
%         output(j,1) = W_h2' * s_h1;
%         
%         error = error + (output(j,1) - trainY(j,1)) * (output(j,1) - trainY(j,1));
%     end
%     
%     error(i,1) = sqrt(error(i,1) / size(trainX,1));
% end
% 
% % plot(trainY,output,'*');
% %figure,plot(error,'*');
% figure,plot(trainX(:,1),output,'*');

[fWithE,tt] = mapminmax(fWithE',-1,1);

W_h1(:,1) = iw(:,1);
W_h1(:,2) = b1(:,1);
W_h2 = lw;
nodes = 10;
output(size(trainX,1),1) = zeros;
error = zeros;
s_h1(nodes,1) = zeros;

for j=1:1:size(trainX,1)
       
%         a_h1 = W_h1 * trainX(j,:)';
        for k=1:1:nodes
            temp = W_h1(k,1) * trainX(j,1) + W_h1(k,2);
            s_h1(k,1) = 1 / (1 + exp(-1 * temp));
        end %s_h1(k,1) = 1 / (1 + exp(-1 .* a_h1(k,1)));
        
        output(j,1) = W_h2' * s_h1 + b2;
%         output(j,1) = s_h1(1,1);
        
        error = error + (output(j,1) - trainY(j,1)) * (output(j,1) - trainY(j,1));
end

figure,plot(trainX(:,1),output,'*');
