function elmModel=uselm2(X,X_test,L,paras,H,H_test)

[N,elmModel.InputDim]=size(X);

% Normalize the input
elmModel.NormalizeInput=paras.NormalizeInput;
if paras.NormalizeInput
    mi = min(X);
    ma = max(X);
    n = size(X,2);
    for i=1:n
        X(:,i) = (X(:,i)-mi(i))/(ma(i)-mi(i));
        X_test(:,i) = (X_test(:,i)-mi(i))/(ma(i)-mi(i));
    end
end

% Random generate input weights
elmModel.InputWeight=rand(elmModel.InputDim,paras.NumHiddenNeuron)*2-1;

% Calculate hidden neuron output matrix
elmModel.Kernel=paras.Kernel;
% switch paras.Kernel
%     case 'sigmoid'
%         H=1 ./ (1 + exp(-X*elmModel.InputWeight));
%         H_test = 1 ./ (1 + exp(-X_test*elmModel.InputWeight));
%     case 'tanh'
%         H= (2 ./ (1 + exp(-2*X*elmModel.InputWeight)))-1;
%         H_test= (2 ./ (1 + exp(-2*X_test*elmModel.InputWeight)))-1;
%     case 'gaussian'
%         H = exp(-(X*elmModel.InputWeight).^2);
%         H_test = exp(-(X_test*elmModel.InputWeight).^2);
%     case 'sinusoid'
%         H = sin(X*elmModel.InputWeight);
%         H_test = sin(X_test*elmModel.InputWeight);
% end

% Calculate output weights
opts.tol = 1e-9;
opts.issym=1;
opts.disp = 0;

if  (paras.NumHiddenNeuron<N)
    A=eye(paras.NumHiddenNeuron)+paras.lambda*H'*L*H;
    B=H'*H;
    [E,V] = eigs(A,B,paras.NE+1,'sm',opts);
    [~,idx]=sort(diag(V));
    elmModel.OutputWeight=E(:,idx(2:end));
    norm_term=H*E(:,idx(2:end));
    elmModel.OutputWeight=bsxfun(@times,E(:,idx(2:end)),sqrt(1./sum(norm_term.*norm_term)));
else
    B=H*H';
    A=eye(N)+paras.lambda*L*B;
    [E,V] = eigs(A,B,paras.NE+1,'sm',opts);
    [~,idx]=sort(diag(V));
    norm_term=B*E(:,idx(2:end));
    elmModel.OutputWeight=bsxfun(@times,H'*E(:,idx(2:end)),sqrt(1./sum(norm_term.*norm_term)));
end

Embed=H*elmModel.OutputWeight;
Embed_test = H_test*elmModel.OutputWeight;

if ~paras.NormalizeOutput
    elmModel.Embed=Embed;
    elmModel.Embed_test=Embed_test;
else
    elmModel.Embed=bsxfun(@times,Embed,1./sqrt(sum(Embed.*Embed,2)));
    elmModel.Embed_test=bsxfun(@times,Embed_test,1./sqrt(sum(Embed_test.*Embed_test,2)));
end


