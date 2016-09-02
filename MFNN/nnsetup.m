function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size(:,1));
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.gamma                            = 0;            
	nn.beta1                            = 0.9;
    nn.beta2                            = 0.999;
    nn.adam                             = 1;
	nn.fullyconnection                  = 1;
    nn.fullyconnectlayer                = 2;
    
    for i = 2 : nn.n 
        % weights and weight momentum
        nn.W1{i - 1} = (rand(nn.size(i,1), nn.size(i - 1,1)) - 0.5) * 2  *4* sqrt(6 / (nn.size(i,1) + nn.size(i - 1,1)));
		nn.W2{i - 1} = (rand(nn.size(i,2), nn.size(i - 1,2)) - 0.5) * 2  *4* sqrt(6 / (nn.size(i,2) + nn.size(i - 1,2)));
        nn.b{i-1}    = (rand(nn.size(i,1), nn.size(i,2))-0.5) * 2 *4* sqrt(12 / (nn.size(i,1) + nn.size(i - 1,1))+(nn.size(i,2) + nn.size(i - 1,2)));
        
        nn.m1{i - 1} = zeros(size(nn.W1{i - 1}));
        nn.m2{i - 1} = zeros(size(nn.W2{i - 1}));
        nn.v1{i - 1} = zeros(size(nn.W1{i - 1}));
        nn.v2{i - 1} = zeros(size(nn.W2{i - 1}));
        nn.mb{i - 1} = zeros(size(nn.b{i - 1}));
        nn.vb{i - 1} = zeros(size(nn.b{i - 1}));
        
		%nn.vW1{i - 1} = zeros(size(nn.W1{i - 1}));
		%nn.vW2{i - 1} = zeros(size(nn.W2{i - 1}));
        
        % average activations (for use with sparsity)
        %nn.p{i}     = zeros(1, nn.size(i));   
    end
	 
	    
end
