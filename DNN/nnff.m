function nn = nnff(nn, x, y , mix, s1,s2)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
	dim = floor(nn.size(n)/2);
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
			case 'ReLU'
                nn.a{i} = max(nn.a{i - 1} * nn.W{i - 1}',0);			
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
	
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
        case 'NN_SCSS'
		    %tmp = sigm(nn.a{n - 1} * nn.W{n - 1}'); %sigmoid
            tmp = nn.a{n - 1} * nn.W{n - 1}'; %linear
			nn.a1 = tmp(:,1:dim);
			nn.a2 = tmp(:,dim+1:end);
            nn.y1= abs(nn.a1)./(abs(nn.a1)+abs(nn.a2)+1e-10);
            nn.y2= abs(nn.a2)./(abs(nn.a1)+abs(nn.a2)+1e-10);
			nn.s1_h = nn.y1.*mix;
			nn.s2_h = nn.y2.*mix;
            nn.a{n} = [nn.s1_h nn.s2_h];
            		
    end

    %error and loss
	switch nn.output
	    case {'sigm', 'linear','softmax'}
            nn.e = y - nn.a{n};
		case 'NN_SCSS'
       		%nn.e = (s1-nn.s1_h) + (s2-nn.s2_h);
            nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum( nn.e.^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
	    case 'NN_SCSS'
		    nn.L = 1/2 * sum(sum( (nn.s1_h-nn.s2_h-s1+s2).^ 2)) / m ;
    end
end
