
clear;

temp3 = dlmread('D:\KMPA\Assignments\Assign1\Assignment1-Data\MultiVariate Dataset\Concrete Compressive Strength Dataset/concrete-data.txt');

b = size(temp3,1);
randoms = randperm(b,b);

biData = temp3(randoms(1,1:round(b * 0.7)),:);
testData = temp3(randoms(1,round(b * 0.7)+1:round(b * 0.9)),:);
dataSize = size(biData,1);
testDataSize = size(testData,1);
dimensionX = size(biData,2) - 1;
 
ZTrain = biData(1:dataSize,3);
ZTest = testData(:,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxGaussians = 598;
steps = 30;
start = 2;
errorTrainArr(:,1) = zeros;
errorTestArr(:,1) = zeros;
index=1;
params(:,1) = zeros;
designTrain(dataSize,:) = zeros;
designTest(testDataSize,:) = zeros;
centres = zeros;
centreDist(size(centres,1),size(centres,1)) = zeros;

for D = start:steps:maxGaussians

[labels,centres] = kmeans(biData,D);

%for finding the deviation of gaussian basis functions
for i=1:1:size(centres,1)

    for j=1:1:size(centres,1)
       
        centreDist(i,j) = norm(centres(i,:) - centres(j,:));
    end
end

%creates covariance matrix
deviation = max(centreDist(:)) / sqrt(2 * D);
temp1(1:dimensionX) = deviation * deviation;
covMatrix = diag(temp1);

%computes design matrix for training data
for i=1:1:dataSize
   
    for j=1:1:D
        
        temp1 = (biData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = (temp1 / covMatrix) *temp1';
        designTrain(i,j) = exp(-1 * numerator / 2);
        temp1 = zeros;
    end
end

designInvTrain = pinv(designTrain);
params = designInvTrain * ZTrain;
modelOutput(1:dataSize,1) = designTrain * params;%finds the model output

temp4 = (modelOutput - ZTrain)' * (modelOutput - ZTrain);
temp4 = temp4 / dataSize;
errorTrain = sqrt(temp4);

errorTrainArr(index,1) = errorTrain;

% figure,
% plot(ZTrain,modelOutput,'*',min(ZTrain):max(ZTrain),min(ZTrain):max(ZTrain));

%%%%%%%%%%%%% test error %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:1:testDataSize
   
    for j=1:1:D
       
        temp2 = (testData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = (temp2 / covMatrix)* temp2';
        designTest(i,j) = exp(-1 * numerator/ 2);
        temp2 = zeros;
    end
    
end

testOutputModel = designTest * params;

temp4 = (ZTest - testOutputModel)' * (ZTest - testOutputModel);
temp4 = temp4 / testDataSize;
testError = sqrt(temp4);
errorTestArr(index,1) = testError;
index = index+1;

end

% errorTrainArr(:,1) = (1/max(errorTestArr)) .* errorTrainArr(:,1);
% errorTestArr(:,1) = (1/max(errorTestArr)) .* errorTestArr(:,1);

%% 

figure,
plot(start:steps:maxGaussians,(errorTrainArr),'-o',start:steps:maxGaussians,(errorTestArr),'-o');
xlabel('No of Gaussian basis funs');
ylabel('Error');
legend('Training error','Validation error');
% hold on
% patch([2 2 6 6],[0 8000 8000 0],'red','LineStyle','none','FaceAlpha',.1);
% hold on
% patch([6 6 10 10],[0 8000 8000 0],'green','LineStyle','none','FaceAlpha',.1);
% hold on
% patch([10 10 20 20],[0 8000 8000 0],'red','LineStyle','none','FaceAlpha',.1);

%figure,plot(1:size(YTrain,1),YTrain,1:size(YTest,1),YTest); % to see the difference between test and validation data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



