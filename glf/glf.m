clear all; clc;
% jpeg90='ms_project\MedianFilterDetection\images\jpeg90\';
% jpeg45='ms_project\MedianFilterDetection\images\jpeg45\';
% m3j45='ms_project\MedianFilterDetection\images\medr3_45\';
% m3j90='ms_project\MedianFilterDetection\images\medr3_90\';
% m5j45='ms_project\MedianFilterDetection\images\medr5_45\';
% m5j90='ms_project\MedianFilterDetection\images\medr5_90\';

% X=[];
% for i=1:100
%     i
%     X =[X spam686(strcat(strcat(m3j90,int2str(i)),'.jpg'))];
% end

%% classification
clear all; clc
load('C:\Users\gps\Documents\MATLAB\ms_project\matlab.mat')
X = [jpeg90(:,1:80) m3j90(:,1:80)]';
y=[];
for i=1:80
    y=[y 0];
end
for i=1:80
    y=[y 1];
end

SVMModel = fitcsvm(X,y);
XTest = [jpeg90(:,81:100) m3j90(:,81:100)]';
yTest=[];
for i=1:20
    yTest=[yTest 0];
end
for i=1:20
    yTest=[yTest 1];
end
yTest=yTest';

[label,score] = predict(SVMModel,XTest);
correct=0;
for i=1:size(yTest,1)
    if label(i) == yTest(i)
        correct = correct + 1;
    end
end

correct/size(yTest,1)