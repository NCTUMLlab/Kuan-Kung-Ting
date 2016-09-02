function [mix_train, mix_test,mix_dev, target1, target2, target1_dev, target2_dev ,target1_test, target2_test ,mix_dev_phase,mix_test_phase] = mix_signal(train1,train2,test1,test2,valid1,valid2,SNRs,L,win,shift,fs);


%normalize to equal power(training)
maxLength=max([length(train1), length(train2)]);
train1(end+1:maxLength)=eps;
train2(end+1:maxLength)=eps;

train1=train1./sqrt(sum(train1.^2));
train2=train2./sqrt(sum(train2.^2));

Px1= train1'* train1/ length(train1);
Px2= train2'* train2/ length(train2);        
sf= sqrt( Px1/Px2/ (10^ (SNRs/ 10)));        
train2= train2 * sf;

% circular shift per 10000 samples 
target1   = [];
target2   = [];
mix_train = [];
sample  = 10000;

for i = 1:sample:length(train2)-sample
     train2_shift = [train2(i: end); train2(1: i-1)];
	 
	 target1 = [target1 , abs(spectrogram(train1,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L))];
	 target2 = [target2 , abs(spectrogram(train2_shift,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L))];
     dmix = train1+ train2_shift;
	 mix_train = [mix_train , abs(spectrogram(dmix,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L))];
end 
% normalize to equal power(validation)
maxLength=max([length(valid1), length(valid2)]);
valid1(end+1:maxLength)=eps;
valid2(end+1:maxLength)=eps;

valid1=valid1./sqrt(sum(valid1.^2));
valid2=valid2./sqrt(sum(valid2.^2));

Px1= valid1'* valid1/ length(valid1);
Px2= valid2'* valid2/ length(valid2);        
sf= sqrt( Px1/Px2/ (10^ (SNRs/ 10)));        
valid2= valid2 * sf;

vmix = valid1 + valid2;
% STFT of validation
target1_dev  = abs(spectrogram(valid1,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));
target2_dev  = abs(spectrogram(valid2,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));
mix_dev      = abs(spectrogram(vmix,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));

mix_dev_phase    = angle(spectrogram(vmix,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));


% normalize to equal power(test)	  
maxLength=max([length(test1), length(test2)]);
test1(end+1:maxLength)=eps;
test2(end+1:maxLength)=eps;

test1=test1./sqrt(sum(test1.^2));
test2=test2./sqrt(sum(test2.^2));

Px1= test1'* test1/ length(test1);
Px2= test2'* test2/ length(test2);        
sf= sqrt( Px1/Px2/ (10^ (SNRs/ 10)));        
test2= test2 * sf;

tmix = test1 + test2;
% STFT of test
target1_test  = abs(spectrogram(test1,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));
target2_test  = abs(spectrogram(test2,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));
mix_test      = abs(spectrogram(tmix,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));

mix_test_phase    = angle(spectrogram(tmix,ceil(win*fs),ceil(win*fs)-ceil(shift*fs), L));

end