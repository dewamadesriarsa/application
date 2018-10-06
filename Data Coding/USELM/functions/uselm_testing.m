function elmModel=uselm_testing(X,model)

[N,elmModel.InputDim]=size(X);

% Normalize the input
if exist('model.PreProcess')==1
    [X]=mapminmax.apply(X,model.PreProcess);
end

% Random generate input weights
elmModel.InputWeight=rand(elmModel.InputDim,paras.NumHiddenNeuron)*2-1;

% Calculate hidden neuron output matrix
elmModel.Kernel=paras.Kernel;
switch paras.Kernel
    case 'sigmoid'
        H=1 ./ (1 + exp(-X*elmModel.InputWeight));
    case 'tanh'
        H= (2 ./ (1 + exp(-2*X*elmModel.InputWeight)))-1;
    case 'gaussian'
        H = exp(-(X*elmModel.InputWeight).^2);
    case 'sinusoid'
        H = sin(X*elmModel.InputWeight);
end

Embed=H*elmModel.OutputWeight;

if ~paras.NormalizeOutput
    elmModel.Embed=Embed;
else
    elmModel.Embed=bsxfun(@times,Embed,1./sqrt(sum(Embed.*Embed,2)));
end


