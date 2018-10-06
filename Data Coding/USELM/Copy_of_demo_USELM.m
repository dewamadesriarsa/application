% Unsupervised ELM (US-ELM) for embedding, dimension reduction and clustering.
% Ref: Huang Gao, Song Shiji, Gupta JND, Wu Cheng, Semi-supervised and
% unsupervised extreme learning machines, IEEE Transactions on Cybernetics, 2014

format compact;
clear; clc;
close all;

addpath(genpath('functions'));
addpath(genpath('classifier'));

% load data
%data=load ('iris.txt');
%data = double(csvread('fold1_iris.csv'));
%data = double(csvread('fold1.csv'));
%data = double(csvread('fold1_ekg.csv'));
%data = double(csvread('fold1_sleep.csv'));
%data = double(csvread('fold1_weka.csv'));
%data = double(csvread('CTG_new.csv'));
data = double(csvread('datasetanwar/mitra-4class-format.csv'));
num_fea = size(data,2)-1;

num_testing = round(length(data)*0.3);

% X=data(:,1:end-1);
% y=data(:,end);

X=data(1:end-num_testing,1:end-1);
y=data(1:end-num_testing,end);
X_test=data(end-num_testing:end,1:end-1);
y_test=data(end-num_testing:end,end);

model = svmtrain(y, X, [, '-c 2']);
[predicted_label] = svmpredict(y_test,X_test, model);

NC=length(unique(y)); % specify number of clusters

% %%%%%%%%%%%%%%%%% Step 1: construct graph Laplacian %%%%%%%%%%%%%%%%%
% hyper-parameter settings for graph
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;
options.NN=5;

L=laplacian(options,X);

% %%%%%%%%%%%%%%%%% Step 2: Run US-ELM for embedding %%%%%%%%%%%%%%%%%%%
% hyper-parameter settings for us-elm
paras.NE=3; % specify dimensions of embedding
%paras.NumHiddenNeuron=2000;
paras.NumHiddenNeuron=2000;
paras.NormalizeInput=1;
paras.NormalizeOutput=1;
%paras.Kernel='sigmoid';
paras.Kernel = 'tanh';
%paras.Kernel = 'gaussian';
%paras.Kernel = 'sinusoid';
paras.lambda=0.1;
elmModel=uselm(X,X_test,L,paras);



% %%%%%%%%%%%%%%%%% Step 3: Run k-means for clustering %%%%%%%%%%%%%%%%%
acc_kmeans=[];acc_le=[];acc_uselm=[];
for i=1:1000
    [label_kmeans, center] = litekmeans(X_test,NC,'MaxIter', 200);
    acc_kmeans(i)=accuracy(y_test,label_kmeans);
    [label_uselm, center] = litekmeans(elmModel.Embed_test, NC, 'MaxIter', 200);
    acc_uselm(i)=accuracy(y_test,label_uselm);
    [idx, C] = kmeans(elmModel.Embed_test,NC);
    acc_uselm_by_matlab(i)=accuracy(y_test,idx);
end

% model = train(y,sparse(elmModel.Embed),[,'s 0']);
% [predicted_label] = predict(y_test, sparse(elmModel.Embed_test), model,[, 's 0']);

model = svmtrain(y, elmModel.Embed, [, '-c 2']);
[predicted_label] = svmpredict(y_test,elmModel.Embed_test, model);

disp(['Clustering accuracy of k-means, Best: ',num2str(max(acc_kmeans)),...
    ' Average: ',num2str(mean(acc_kmeans))]);
disp(['Clustering accuracy of US-ELM, Best: ',num2str(max(acc_uselm)),...
    ' Average: ',num2str(mean(acc_uselm))]);
disp(['Clustering accuracy of MATLAB, Best: ',num2str(max(acc_uselm_by_matlab)),...
    ' Average: ',num2str(mean(acc_uselm_by_matlab))]);

% %%%%%%%%%%%%%%%%%%%%% 3-D plot of the results %%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% E=X;
% hold on
% title('The original IRIS data')
% view(3)
% plot3(E(y==1,1),E(y==1,2),E(y==1,3),'gx','MarkerSize',8,'LineWidth',1.5)
% plot3(E(y==2,1),E(y==2,2),E(y==2,3),'c+','MarkerSize',6,'LineWidth',1.5)
% plot3(E(y==3,1),E(y==3,2),E(y==3,3),'b.','MarkerSize',10,'LineWidth',1.5)
% grid on
% axis square
% 
% 
% figure(2)
% E=elmModel.Embed;
% hold on
% title('The embedded IRIS data')
% view(3)
% plot3(E(y==1,1),E(y==1,2),E(y==1,3),'gx','MarkerSize',8,'LineWidth',1.5)
% plot3(E(y==2,1),E(y==2,2),E(y==2,3),'c+','MarkerSize',6,'LineWidth',1.5)
% plot3(E(y==3,1),E(y==3,2),E(y==3,3),'b.','MarkerSize',10,'LineWidth',1.5)
% grid on
% axis square
