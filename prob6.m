
clear;

 biData = dlmread('D:/KMPA/Assignments/Assign1/Assignment1-Data/2.Bivariate/group5/bivariateData/group5_train1000.txt');
 %biData = dlmread('/home/cs16m034/Documents/KMPA/Assignments/1/Assignment1-Data/MultiVariate Dataset/Concrete Compressive Strength Dataset/concrete-data.txt');
 testData = dlmread('D:/KMPA/Assignments/Assign1/Assignment1-Data/2.Bivariate/group5/bivariateData/group5_val.txt');
 dataSize = size(biData,1);
 dimensionX = size(biData,2) - 1;
 
XTrain = biData(1:dataSize,1);
YTrain = biData(1:dataSize,2);
ZTrain = biData(1:dataSize,3);

XTest = testData(:,1);
YTest = testData(:,2);
ZTest = testData(:,3);

minX = min(XTrain);
maxX = max(XTrain);
minY = min(YTrain);
maxY = max(YTrain);

[Xx,Yy] = meshgrid(minX:0.4:maxX,minY:0.4:maxY);

ZzTrain = griddata(XTrain,YTrain,ZTrain,Xx,Yy);

% figure
% mesh(Xx,Yy,Zz);
% hold on
% plot3(X,Y,Z,'o');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxGaussians = 80;
startMixture = 5;
errorTrainArr(1:(maxGaussians-startMixture+1),1) = zeros;
errorTestArr(1:(maxGaussians-startMixture+1),1) = zeros;
index=1;

for D = startMixture:1:maxGaussians
  
% centreIndex = randperm(dataSize,D);
% centres = biData(centreIndex,:);

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
        numerator = temp2 * inv(covMatrix) * temp2';
        phiMatrix(i,j) = exp(-1 * numerator / 2);
    end
end

params(D,1) = zeros;

designTrain(dataSize,D) = zeros;

%computes design matrix for training data
for i=1:1:dataSize
   
    for j=1:1:D
        
        temp = (biData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = temp * inv(covMatrix) * temp';
        designTrain(i,j) = exp(-1 * numerator / 2);
    end
    
end

lamda = exp(-6);
temp3 = lamda .* phiMatrix;
designInvTrain = inv(designTrain' * designTrain + temp3);

params = designInvTrain * designTrain' * ZTrain;

modelOutput(1:dataSize,1) = designTrain * params;%finds the model output

errorTrain = norm(modelOutput - ZTrain);

errorTrainArr(index,1) = errorTrain;


%ZzModel = griddata(XTrain,YTrain,modelOutput,Xx,Yy);



% figure
% hSurface = surf(Xx,Yy,ZzTrain);
% set(hSurface, 'FaceColor',[1 0 0], 'FaceAlpha',0.7, 'EdgeAlpha', 0);
% % hold on
% % plot3(X,Y,Z,'o');
% hold on
% mSurf = surf(Xx,Yy,ZzModel);
% set(mSurf, 'FaceColor',[0 0 1], 'FaceAlpha',0.7, 'EdgeAlpha', 0);


% figure,
% plot(ZTrain,modelOutput,'*',min(ZTrain):max(ZTrain),min(ZTrain):max(ZTrain));



%%%%%%%%%%%%% test error %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


testDataSize = size(testData,1);
designTest(testDataSize,D) = zeros;

for i=1:1:testDataSize
   
    for j=1:1:D
       
        temp = (testData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = temp * temp';
        designTest(i,j) = exp(numerator/(2 * deviation * deviation));
    end
    
end

testOutputModel = designTest * params;

testError = norm(ZTest - testOutputModel);

errorTestArr(index,1) = testError;
index = index+1;

end

% errorTrainArr(:,1) = (1/max(errorTestArr)) .* errorTrainArr(:,1);
% errorTestArr(:,1) = (1/max(errorTestArr)) .* errorTestArr(:,1);

figure,
plot(startMixture:maxGaussians,errorTrainArr,'o',startMixture:(maxGaussians),errorTestArr,'o');
hold on
plot(startMixture:maxGaussians,errorTrainArr,startMixture:(maxGaussians),errorTestArr);

[XxTest,YyTest] = meshgrid(min(XTest):0.4:max(XTest),min(YTest):0.4:max(YTest));
ZzTestModel = griddata(XTest,YTest,testOutputModel,XxTest,YyTest);
ZzTest = griddata(XTest,YTest,ZTest,XxTest,YyTest);

% figure
% hSurface = surf(XxTest,YyTest,ZzTestModel);
% set(hSurface, 'FaceColor',[1 0 0], 'FaceAlpha',0.7, 'EdgeAlpha', 0);
% hold on
% testSurface = surf(XxTest,YyTest,ZzTest);
% set(testSurface, 'FaceColor',[0 0 1], 'FaceAlpha',0.7, 'EdgeAlpha', 0);

%figure,plot(1:size(YTrain,1),YTrain,1:size(YTest,1),YTest); % to see the difference between test and validation data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%