 function [nn, L]  = nntrain(nn, train_x, train_y, mix, s1,s2,dim,opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.
% mix means mixed signal in current frame
% dim means dimension in frequency bin 
assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 8 || nargin == 10,'number of input arguments must be 8 or 10')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 10
    opts.validation = 1;
end


fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 3);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
% time step for Adam grad.
t = 0; 
for i = 1 : numepochs
    tic;
    kk = randperm(m);
    for l = 1 : numbatches
        L(n) = 0;
        t = t+1;
        for lay = 2 : nn.n 
        nn.dW1{lay-1} = zeros(size(nn.W1{lay - 1})); 
		nn.dW2{lay-1} = zeros(size(nn.W2{lay - 1})); 
        nn.db{lay-1} = zeros(size(nn.b{lay-1}));    
        end
       
        for batch = 1 : batchsize
            
         batch_x = train_x(:,:,kk((l - 1) * batchsize + batch));
         batch_mix = mix(:,kk((l - 1) * batchsize + batch));
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
       batch_y = train_y(:,kk((l - 1) * batchsize + batch));

		batch_s1 = batch_y(1:dim,:);
		batch_s2 = batch_y(dim+1:end,:);
        
        nn = nnff(nn, batch_x, batch_y, batch_mix, batch_s1, batch_s2);
        nn = nnbp(nn, batch_mix, batch_s1, batch_s2); 
        L(n) = L(n) + nn.L;
        end
      
        nn = nnapplygrads(nn,batchsize,t);
        L(n) = L(n)/batchsize;
        n = n + 1;
    end
    
    t = toc;

%     if opts.validation == 1
%         loss = nneval(nn, loss, train_x, train_y,mix,s1,s2,val_x, val_y);
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%     else
%         loss = nneval(nn, loss, train_x, train_y,mix,s1,s2);
%         str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
%     end
%     if ishandle(fhandle)
%         nnupdatefigures(nn, fhandle, loss, opts, i);
%     end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);% str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end

