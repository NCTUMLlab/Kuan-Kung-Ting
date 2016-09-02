clear all;
clc;
%% data prepared
SNRs=0;
[train1 , fs] = wavread('..\Data\female_train.wav');
[train2 , fs] = wavread('..\Data\male_train.wav');
[valid1 , fs] = wavread('..\Data\female_dev.wav');
[valid2 , fs] = wavread('..\Data\male_dev.wav');
[test1 , fs]  = wavread('..\Data\female_test.wav');
[test2 , fs]  = wavread('..\Data\male_test.wav');

%window setting
%number of fft point
numfft = 1024;
%window size(in second)
win = 0.064;
%overlap size(in second)
shift = 0.032;

%data prepare
[mix_train, mix_test, mix_dev,target1, target2, target1_dev, target2_dev ,target1_test, target2_test ,mix_dev_phase,mix_test_phase] = mix_signal(train1,train2,test1,test2,valid1,valid2,SNRs,numfft,win,shift,fs);


 train_frame_size  = length(mix_train(1,:));
 train_frq_size    = length(mix_train(:,1));
 test_frame_size   = length(mix_test(1,:));
 
% data with frame n
shift_size = 1;
% training data
mix_train_add = [mix_train(:,((train_frame_size)-shift_size+1):(end)) mix_train mix_train(:,1:shift_size)]; 
X =[];
for i = 1:train_frame_size
    X(:,:,i) = mix_train_add(:,i:(i+2*shift_size));
end
% dev data
mix_dev_add = [mix_dev(:,(length(mix_dev(1,:))-shift_size+1):(end)) mix_dev mix_dev(:,1:shift_size)]; 
X_dev =[];
for i = 1:length(mix_dev(1,:))
     X_dev(:,:,i) = mix_dev_add(:,i:(i+2*shift_size));
end
% test data
mix_test_add = [mix_test(:,(test_frame_size-shift_size+1):(end)) mix_test mix_test(:,1:shift_size)]; 
X_test =[];
for i = 1:test_frame_size
     X_test(:,:,i) = mix_test_add(:,i:(i+2*shift_size));
end

y_target = [target1 ; target2];
context_window = 2*shift_size+1;
dim            = train_frq_size; 
%% load layerwise pre-trained weight 
%load 'MFNN-3-150-3-150-3-150-3.mat'
% Weight1 = nn.W1{1};
% Weight2 = nn.W2{1};
% bias = nn.b{1};
% Weight1_2 = nn.W1{2};
% Weight2_2 = nn.W2{2};
% bias_2 = nn.b{2};
% Weight1_3 = nn.W1{3};
% Weight2_3 = nn.W2{3};
% bias_3 = nn.b{3};
% Weight1_4 = nn.W1{4};
% Weight2_4 = nn.W2{4};
% bias_4 = nn.b{4};
% Weight1_5 = nn.W1{5};
% Weight2_5 = nn.W2{5};
% bias_5 = nn.b{5};
% Weight1_6 = nn.W1{6};
% Weight2_6 = nn.W2{6};
% bias_6 = nn.b{6};
%% training
opts.batchsize  = 50;
opts.numepochs  = 50;
loss_fig   = [];

nn   =  nnsetup([dim,context_window;150,3;150,3;450,1;450,1;450,1;450,1;dim*2,1]);
% If layerwise pre-train, load weight%
% nn.W1{1} = Weight1;
% nn.W2{1} = Weight2;
% nn.b{1}  = bias;
% nn.W1{2} = Weight1_2;
% nn.W2{2} = Weight2_2;
% nn.b{2}  = bias_2;
% nn.W1{3} = Weight1_3;
% nn.W2{3} = Weight2_3;
% nn.b{3}  = bias_3;
% nn.W1{4} = Weight1_4;
% nn.W2{4} = Weight2_4;
% nn.b{4}  = bias_4;
% nn.W1{5} = Weight1_5;
% nn.W2{5} = Weight2_5;
% nn.b{5}  = bias_5;
% nn.W1{6} = Weight1_6;
% nn.W2{6} = Weight2_6;
% nn.b{6}  = bias_6;

nn.activation_function              = 'ReLU';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 0.001;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum                         = 0;          %  Momentum
nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
nn.weightPenaltyL2                  = 0;            %  L2 regularization
nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
nn.sparsityTarget                   = 0;         %  Sparsity target
nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
nn.output                           = 'NN_SCSS';    %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
nn.adam                             = 1;            %  1 for adam grad.
nn.fullyconnection                  = 1;            %  1 for add fully-connected layer
nn.fullyconnectlayer                = 4;            % number of fully-connected layer

[nn, L] = nntrain(nn, X(:,:,1:26700), y_target(:,1:26700),mix_train(:,1:26700),target1(:,1:26700),target2(:,1:26700),dim,opts);
%% modelname
modelname = ['model_MFNN','_',nn.activation_function,'_win',num2str(context_window),'_L',num2str(nn.n-2)];
if nn.fullyconnection ==1
    modelname = [modelname,'_h',num2str(nn.size(2,1)),'x',num2str(nn.size(2,2)),'_',num2str(nn.size(nn.n-nn.fullyconnectlayer,1)),'_fullyL',num2str(nn.fullyconnectlayer)];
else
    modelname = [modelname,'_h',num2str(nn.size(2,1)),'x',num2str(nn.size(2,2)),'_nonfully'];
end

if nn.adam == 1
    modelname = [modelname,'_AdamSGD'];
else
    modelname = [modelname,'_SGD'];
end
%% test
% test set
y_target_test = [target1_test ; target2_test];

for i =1:test_frame_size
    nnout = nnff(nn, X_test(:,:,i) , y_target_test(:,i) , mix_test(:,i) , target1_test(:,i), target2_test(:,i));
    source1_t(:,i) = nnout.s1_h;
    source2_t(:,i) = nnout.s2_h;
end

source1_test = OLA(source1_t,mix_test_phase,win,shift,fs);
source2_test = OLA(source2_t,mix_test_phase,win,shift,fs);

%equal length
maxLength=max([length(test1), length(test2)]);
test1(end+1:maxLength)=eps;
test2(end+1:maxLength)=eps;

test1=test1./sqrt(sum(test1.^2));
test2=test2./sqrt(sum(test2.^2));

Px1= test1'* test1/ length(test1);
Px2= test2'* test2/ length(test2);        
sf= sqrt( Px1/Px2/ (10^ (SNRs/ 10)));        
test2= test2 * sf;
%calculate measures
src1 = [source1_test;source2_test];
src2 = [test1(1:length(source1_test))';test2(1:length(source2_test))'];
[SDR_test,SIR_test,SAR_test] = bss_eval_sources(src1,src2);

% dev set 
y_target_dev = [target1_dev ; target2_dev];

for i=1:length(mix_dev(1,:));
 nnout = nnff(nn, X_dev(:,:,i) , y_target_dev(:,i) , mix_dev(:,i) , target1_dev(:,i), target2_dev(:,i));
 source1(:,i) = nnout.s1_h;
 source2(:,i) = nnout.s2_h;
end
source1_dev = OLA(source1,mix_dev_phase,win,shift,fs);
source2_dev = OLA(source2,mix_dev_phase,win,shift,fs);
%equal length
maxLength=max([length(valid1), length(valid2)]);
valid1(end+1:maxLength)=eps;
valid2(end+1:maxLength)=eps;

valid1=valid1./sqrt(sum(valid1.^2));
valid2=valid2./sqrt(sum(valid2.^2));

Px1= valid1'* valid1/ length(valid1);
Px2= valid2'* valid2/ length(valid2);        
sf= sqrt( Px1/Px2/ (10^ (SNRs/ 10)));        
valid2= valid2 * sf;
%calculate measures
src1 = [source1_dev;source2_dev];
src2 = [valid1(1:length(source1_dev))';valid2(1:length(source2_dev))'];
[SDR_dev,SIR_dev,SAR_dev] = bss_eval_sources(src1,src2);
% write to file
fp =  fopen('result_mfnn.txt', 'a');
fprintf(fp, '%s\ttestSDR:\t%.3f\ttestSIR:\t%.3f\ttestSAR:\t%.3f\n', modelname, mean(SDR_test), mean(SIR_test), mean(SAR_test));
fprintf(fp, '%s\tdevSDR:\t%.3f\tdevSIR:\t%.3f\tdevSAR:\t%.3f\n', modelname, mean(SDR_dev), mean(SIR_dev), mean(SAR_dev));
fclose(fp);

%%
filename = ['..\Record\',modelname];
testS1 = [filename,'_testS1.wav'];
testS2 = [filename,'_testS2.wav'];
devS1 = [filename,'_devS1.wav'];
devS2 = [filename,'_devS2.wav'];
wavwrite(source1_test,fs,testS1);
wavwrite(source2_test,fs,testS2);
wavwrite(source1_dev,fs,devS1);
wavwrite(source2_dev,fs,devS2);




