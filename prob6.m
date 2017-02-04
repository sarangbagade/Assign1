
clear;

 biData = dlmread('D:/KMPA/Assignments/Assign1/Assignment1-Data/2.Bivariate/group5/bivariateData/group5_train1000.txt');
 testData = dlmread('D:/KMPA/Assignments/Assign1/Assignment1-Data/2.Bivariate/group5/bivariateData/group5_val.txt');
 trainDataSize = size(biData,1);
 dimensionX = size(biData,2) - 1;
 
ZTrain = biData(1:trainDataSize,3);
ZTest = testData(:,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = 20;

[labels,centres] = kmeans(biData,D);

centreDist(size(centres,1),size(centres,1)) = zeros;

phiMatrix(1:D,1:D) = zeros;

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

% to find phi tild
for i=1:1:size(centres,1)

    for j=1:1:size(centres,1)
               
        temp2 = (centres(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = (temp2 / covMatrix) * temp2';
%         numerator = temp2 * temp2';
%         phiMatrix(i,j) = exp(-1 * numerator / (2 * deviation * deviation));
        phiMatrix(i,j) = exp(-1 * numerator / 2);
        temp2 = zeros;
    end
end

params(D,1) = zeros;
designTrain(trainDataSize,D) = zeros;

%computes design matrix for training data
for i=1:1:trainDataSize
   
    for j=1:1:D
        
        temp3 = (biData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = (temp2 / covMatrix) * temp2';
%         numerator = temp3 * temp3';
%         designTrain(i,j) = exp(-1 * numerator / (2 * deviation * deviation));
        designTrain(i,j) = exp(-1 * numerator / 2);
        temp3 = zeros;
    end
end

testDataSize = size(testData,1);
designTest(testDataSize,D) = zeros;

for i=1:1:testDataSize
   
    for j=1:1:D
       
        temp = (testData(i,1:dimensionX) - centres(j,1:dimensionX));
        %numerator = (temp2 / covMatrix) * temp2';
        numerator = temp * temp';
        designTest(i,j) = exp(-1 * numerator / (2 * deviation * deviation));
        %designTest(i,j) = exp(-1 * numerator / 2);
        temp = zeros;
    end
end

lamdaArr(:,1) = exp(-15:0.5:4);
lamdaArr(1,1) = 0;
index=1;
errorTrainArr(1:size(lamdaArr,1),1) = zeros;
errorTestArr(1:size(lamdaArr,1),1) = zeros;

for k=1:1:size(lamdaArr,1)

lamda = lamdaArr(k,1);
temp5 = designTrain' * designTrain + lamda .* phiMatrix;

params = (temp5 \ designTrain') * ZTrain;

modelOutput(1:trainDataSize,1) = designTrain * params;%finds the model output

errorTrain = norm(modelOutput - ZTrain);

errorTrainArr(index,1) = errorTrain;

%%%%%%%%%%%%% test error %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testOutputModel = designTest * params;

testError = norm(testOutputModel - ZTest);

errorTestArr(index,1) = testError;
index = index+1;

params = zeros;

end

figure,
plot(-15:0.5:4,(errorTrainArr),'-o',-15:0.5:4,(errorTestArr),'-o');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%