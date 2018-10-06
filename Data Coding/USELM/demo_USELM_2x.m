% Unsupervised ELM (US-ELM) for embedding, dimension reduction and clustering.
% Ref: Huang Gao, Song Shiji, Gupta JND, Wu Cheng, Semi-supervised and
% unsupervised extreme learning machines, IEEE Transactions on Cybernetics, 2014

format compact;
clear; clc;
close all;

addpath(genpath('functions'));
addpath(genpath('classifier'));

num_fold = 5;
for i=1:num_fold
    disp(strcat('Fold ',num2str(i)));
    
%     %%% Dataset Skew Sintetis
    X_train = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-training.csv')));
    X_test = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-test.csv')));
    
    %%% Dataset Skew Sintetis
%     X_train = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-2fold-train.csv')));
%     X_test = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-2fold-test.csv')));

%     %%% Dataset Regular Sintetis
%     X_train = double(csvread(strcat('dataset_sintetis/regular_overlap/fold',num2str(i),'-regular_synthetic_5fold-training.csv')));
%     X_test = double(csvread(strcat('dataset_sintetis/regular_overlap/fold',num2str(i),'-regular_synthetic_5fold-test.csv')));
    
%     X_train = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-glass-training.csv')));
%     X_test = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-glass-test.csv')));
    
    num_fea = size(X_train,2)-1;

    X=X_train(:,1:end-1);
    y_train=X_train(:,end);
    y_test=X_test(:,end);
    X_test=X_test(:,1:end-1);
%     
%     mi = min(X);
%     ma = max(X);
% 
%     for j=1:size(X,2)
%         x_norm(:,j) = (X(:,j)-mi(j))/(ma(j)-mi(j));
%         xt_norm(:,j) = (X_test(:,j)-mi(j))/(ma(j)-mi(j));
%     end
%    
%     X = x_norm;
%     X_test = xt_norm;
%     clear x_norm xt_norm;
    
    NC=length(unique(y_train)); % specify number of clusters
    model = svmtrain(y_train,X, [, '-t 2','-h 0']);
    [predicted_label,acc_nonUSELM, decision_values] = svmpredict(y_test,X_test, model);
    
    acc_nonUSELM_rbf_kernel(i) = acc_nonUSELM(1);
    
    model = svmtrain(y_train,X, [, '-t 0','-h 0']);
    [predicted_label,acc_nonUSELM, decision_values] = svmpredict(y_test,X_test, model);
    
    acc_nonUSELM_linear(i) = acc_nonUSELM(1);

    % %%%%%%%%%%%%%%%%% Step 1: construct graph Laplacian %%%%%%%%%%%%%%%%%
    % hyper-parameter settings for graph
    
    options.GraphWeights='binary';
    options.GraphDistanceFunction='euclidean';
    options.LaplacianNormalize=0;
    options.LaplacianDegree=1;
    options.NN=5;

    L=laplacian(options,X);

    disp('ELM training');
    % %%%%%%%%%%%%%%%%% Step 2: Run US-ELM for embedding %%%%%%%%%%%%%%%%%%%
    % hyper-parameter settings for us-elm
    paras.NE=15; 
    paras.NumHiddenNeuron=2000;
    paras.NormalizeInput=1;
    paras.NormalizeOutput=1;
    paras.Kernel='sigmoid';
    %paras.Kernel = 'tanh';
    %paras.Kernel = 'gaussian';
    %paras.Kernel = 'sinusoid';
    paras.lambda=0.1;
    elmModel=uselm(X,X_test,L,paras);
    
    L=laplacian(options,elmModel.Embed);

    disp('ELM training');
    % %%%%%%%%%%%%%%%%% Step 2: Run US-ELM for embedding %%%%%%%%%%%%%%%%%%%
    % hyper-parameter settings for us-elm
    paras2.NE=15; 
    paras2.NumHiddenNeuron=2000;
    paras2.NormalizeInput=1;
    paras2.NormalizeOutput=1;
    paras2.Kernel='sigmoid';
    %paras.Kernel = 'tanh';
    %paras.Kernel = 'gaussian';
    %paras.Kernel = 'sinusoid';
    paras2.lambda=0.1;
    elmModel2=uselm(elmModel.Embed,elmModel.Embed_test,L,paras);
% %
    

    % model = train(y,sparse(elmModel.Embed),[,'s 0']);
    % [predicted_label] = predict(y_test, sparse(elmModel.Embed_test), model,[, 's 0']);
    disp('Train SVM');
    model2 = svmtrain(y_train, elmModel2.Embed, [, '-t 0','-h 0']);
    [predicted_label,accuracy1, decision_values] = svmpredict(y_test,elmModel2.Embed_test, model2);
    
    acc_USELM_linear(i) = accuracy1(1);
    
    model2 = svmtrain(y_train, elmModel2.Embed, [, '-t 2','-h 0']);
    [predicted_label,accuracy1, decision_values] = svmpredict(y_test,elmModel2.Embed_test, model2);
    
    acc_USELM_rbf_kernel(i) = accuracy1(1);
%  
%     for j=1:1000
%         [label_kmeans, center] = litekmeans(X_test,NC,'MaxIter', 200);
%         acc_kmeans(i,j)=accuracy(y_test,label_kmeans);
%         [label_uselm, center] = litekmeans(elmModel.Embed_test, NC, 'MaxIter', 200);
%         acc_uselm(i,j)=accuracy(y_test,label_uselm);
%         [idx, C] = kmeans(elmModel.Embed_test,NC);
%         acc_uselm_by_matlab(i,j)=accuracy(y_test,idx);
%     end

end
% mean_all(1,1) = mean(acc_nonUSELM_linear);
% mean_all(1,2) = mean(acc_nonUSELM_rbf_kernel);
% mean_all(1,3) = mean(acc_USELM_linear);
% mean_all(1,4) = mean(acc_USELM_rbf_kernel);
% 
% mean_all(2,1) = std(acc_nonUSELM_linear);
% mean_all(2,2) = std(acc_nonUSELM_rbf_kernel);
% mean_all(2,3) = std(acc_USELM_linear);
% mean_all(2,4) = std(acc_USELM_rbf_kernel);
