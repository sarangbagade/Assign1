
clear;

x = rand([100 1]);

vari = 0.1;
error = 0 + sqrt(vari) .* randn(100,1); % Error with 0 mean and 0.1 variance
%error = error';

%figure,plot(1:100,randn([1 100]),'*');

funct = exp(cos(2*pi*x)) + x; % Input function

trueTarget = funct;

fWithE = funct + error;

% fid = fopen('Input.txt','wt');
% fprintf(fid,'%1.6f\n',x);
% fclose(fid);
% 
% fid = fopen('ModelOutput.txt','wt');
% fprintf(fid,'%1.6f\n',fWithE);
% fclose(fid);
% 
% fid = fopen('TrueOutput.txt','wt');
% fprintf(fid,'%1.6f\n',trueTarget);
% fclose(fid);

% figure,plot(x,fWithE,'*',x,trueTarget,'*');




