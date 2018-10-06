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
    %%% ECG SLEEP
%     X_train = double(csvread(strcat('datasetanwar/fold',num2str(i),'-mitra-training.csv')));
%     X_test = double(csvread(strcat('datasetanwar/fold',num2str(i),'-mitra-test.csv')));
    
    %%% ODOR 16F
%     X_train = double(csvread(strcat('datasetanwar/fold',num2str(i),'-odor-16f-training.csv')));
%     X_test = double(csvread(strcat('datasetanwar/fold',num2str(i),'-odor-16f-test.csv')));

%     %%% ODOR 8F
%     X_train = double(csvread(strcat('datasetanwar/fold',num2str(i),'-odor-8f-training.csv')));
%     X_test = double(csvread(strcat('datasetanwar/fold',num2str(i),'-odor-8f-test.csv')));
    
%     %%% Dataset Skew Sintetis
    X_train = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-training.csv')));
    X_test = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-test.csv')));
    
    %%% Dataset Skew Sintetis
%     X_train = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-2fold-train.csv')));
%     X_test = double(csvread(strcat('dataset_sintetis/skew_overlap/fold',num2str(i),'-skew-3class-2fold-test.csv')));

%     %%% Dataset Regular Sintetis
%     X_train = double(csvread(strcat('dataset_sintetis/regular_overlap/fold',num2str(i),'-regular_synthetic_5fold-training.csv')));
%     X_test = double(csvread(strcat('dataset_sintetis/regular_overlap/fold',num2str(i),'-regular_synthetic_5fold-test.csv')));
    
    %%% Dataset Regular Sintetis
%     X_train = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-mitra-3class-training.csv')));
%     X_test = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-mitra-3class-test.csv')));
    
%     X_train = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-glass-training.csv')));
%     X_test = double(csvread(strcat('dataset_ecg_sleep/fold',num2str(i),'-glass-test.csv')));
    
    num_fea = size(X_train,2)-1;

    X=X_train(:,1:end-1);
    y_train=X_train(:,end);
    y_test=X_test(:,end);
    X_test=X_test(:,1:end-1);
    
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
    paras.lambda=0.1;
    elmModel=uselm(X,X_test,L,paras);
    
    all_elmModel{i,1} = elmModel;
    
    disp('Train SVM');
    model2 = svmtrain(y_train, elmModel.Embed, [, '-t 0','-h 0']);
    [predicted_label,accuracyL1, decision_values] = svmpredict(y_test,elmModel.Embed_test, model2);
    
    acc_USELM_linear(i,1) = accuracyL1(1);
    all_classifier_linear{i,1} = model2;
    
    model2 = svmtrain(y_train, elmModel.Embed, [, '-t 2','-h 0']);
    [predicted_label,accuracyR1, decision_values] = svmpredict(y_test,elmModel.Embed_test, model2);
    
    acc_USELM_rbf_kernel(i,1) = accuracyR1(1);
    all_classifier_rbf{i,1} = model2;
    
    for k=2:10
        L=laplacian(options,elmModel.Embed);
        paras2.NE=15; 
        paras2.NumHiddenNeuron=2000;
        paras2.NormalizeInput=1;
        paras2.NormalizeOutput=1;
        paras2.Kernel='sigmoid';
        paras2.lambda=0.1;
        elmModel2=uselm(elmModel.Embed,elmModel.Embed_test,L,paras);
        
        model2 = svmtrain(y_train, elmModel2.Embed, [, '-t 0','-h 0']);
        [predicted_label,accuracyL, decision_values] = svmpredict(y_test,elmModel2.Embed_test, model2);
        
        model3 = svmtrain(y_train, elmModel2.Embed, [, '-t 2','-h 0']);
        [predicted_label,accuracyR, decision_values] = svmpredict(y_test,elmModel2.Embed_test, model3);
        
        if(accuracyL1(1) <= accuracyL(1))
            acc_USELM_linear(i,k) = accuracyL(1);
            all_classifier_linear{i,k} = model2;
            
            acc_USELM_rbf_kernel(i,k) = accuracyR(1);
            all_classifier_rbf{i,k} = model3;

            all_elmModel{i,k} = elmModel2;
            elmModel = elmModel2;
            accuracyL1(1) = accuracyL(1);
        else
            break;
        end
    end
    accuracyL
    accuracyR
    
%     L=laplacian(options,elmModel.Embed);
% 
%     disp('ELM training');
    % %%%%%%%%%%%%%%%%% Step 2: Run US-ELM for embedding %%%%%%%%%%%%%%%%%%%
    % hyper-parameter settings for us-elm
%     paras2.NE=9; 
%     paras2.NumHiddenNeuron=2000;
%     paras2.NormalizeInput=1;
%     paras2.NormalizeOutput=1;
%     paras2.Kernel='sigmoid';
%     paras.Kernel = 'tanh';
%     paras.Kernel = 'gaussian';
%     paras.Kernel = 'sinusoid';
%     paras2.lambda=0.1;
%     elmModel2=uselm(elmModel.Embed,elmModel.Embed_test,L,paras);
    

    % model = train(y,sparse(elmModel.Embed),[,'s 0']);
    % [predicted_label] = predict(y_test, sparse(elmModel.Embed_test), model,[, 's 0']);

end