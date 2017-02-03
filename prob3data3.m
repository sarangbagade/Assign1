
clear;

% x(:,1) = -10:10;
% y(:,1) = 1 ./ (1+exp(-0.5 * (x - 2)));
% plot(x,y);

 tempData = dlmread('/home/cs16m034/Documents/KMPA/Assignments/1/Assignment1-Data/MultiVariate Dataset/Concrete Compressive Strength Dataset/concrete-data.txt');
 tempDataSize = size(tempData,1);
 dataSize = abs(tempDataSize * 0.7);
 testDataSize = tempDataSize - dataSize;
 dataIndex = randperm(tempDataSize,tempDataSize);
 testData = tempData(dataIndex(dataSize+1:tempDataSize),:);
 biData = tempData(dataIndex(1:dataSize),:);
 
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

maxGaussians = 20;
errorTrainArr(1:(maxGaussians-1),1) = zeros;
errorTestArr(1:(maxGaussians-1),1) = zeros;
index=1;

for D = 2:1:maxGaussians
    
% centreIndex = randperm(dataSize,D);
% centres = biData(centreIndex,:);

[labels,centres] = kmeans(biData,D);

centreDist(size(centres,1),size(centres,1)) = zeros;

%for finding the deviation of gaussian basis functions
for i=1:1:size(centres,1)

    for j=1:1:size(centres,1)
       
        centreDist(i,j) = norm(centres(i,:) - centres(j,:));
    end
end

%deviation = max(centreDist(:));
deviation = 50;
params(D,1) = zeros;

designTrain(dataSize,D) = zeros;

%compute design matrix for training data
for i=1:1:dataSize
   
    for j=1:1:D
        
        temp = (biData(i,1:dimensionX) - centres(j,1:dimensionX));
        numerator = temp * temp';
        designTrain(i,j) = exp(-1 * numerator/(2 * deviation * deviation));
    end
    
end

designInvTrain = pinv(designTrain);

params = designInvTrain * ZTrain;

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

errorTrainArr(:,1) = (1/max(errorTestArr)) .* errorTrainArr(:,1);
errorTestArr(:,1) = (1/max(errorTestArr)) .* errorTestArr(:,1);

figure,
plot(2:maxGaussians,errorTrainArr,'o',2:(maxGaussians),errorTestArr,'o');
hold on
plot(2:maxGaussians,errorTrainArr,2:(maxGaussians),errorTestArr);

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
