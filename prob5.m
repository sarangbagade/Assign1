% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by Neural Fitting app
% Created 09-Feb-2017 16:48:58
%
% This script assumes these variables are defined:
%
%   input - input data.
%   ZTrain - target data.
load('biDataprob5.mat');
x = input';
t = ZTrain';

minX = min(x(1,:));
maxX = max(x(1,:));
minY = min(x(2,:));
maxY = max(x(2,:));

[Xx,Yy] = meshgrid(minX:0.4:maxX,minY:0.4:maxY);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenL1Size = 30;
hiddenL2Size = 30;

net = fitnet([hiddenL1Size hiddenL2Size],trainFcn);

net.numLayers = 3;

% net.layers{2}.size = hiddenL2Size;
net.layers{2}.transferFcn = 'tansig';
net.layers{2}.initFcn = 'initnw';

view(net);
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,x,t);
net.outputConnect = [1 1 1];

% Test the Network
y = net(x);

view(net);

%%
figure,
for i=1:round(hiddenL1Size/2):hiddenL1Size
    
ZzModel = griddata(x(1,:),x(2,:),y(i,:),Xx,Yy);

subplot(1,3,i) 
mSurf = surf(Xx,Yy,ZzModel);
set(mSurf, 'edgecolor','none');
% zlabel('Z axis')
% xlabel('X axis')
% ylabel('Y axis')

end

%%
ZzModel = griddata(x(1,:),x(2,:),y(61,:),Xx,Yy);

figure,
scatter3(x(1,:),x(2,:),t);
hold on
mSurf = surf(Xx,Yy,ZzModel);
set(mSurf, 'edgecolor','none');

%%
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net);

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
