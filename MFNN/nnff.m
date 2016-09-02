function nn = nnff(nn, x, y , mix, s1,s2)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)
global  sizete1 sizete2;
    n = nn.n;
	dim = floor(nn.size(n,:)/2);
    nn.a{1} = x;
    
    if (nn.fullyconnection== 1)
       l = n-nn.fullyconnectlayer+1;
    else
       l = n;
    end
    
    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                %nn.a{i} = sigm(nn.W1{i-1} * nn.a{i - 1} * nn.W2{i - 1}' + nn.b{i-1});
                nn.a{i} = sigm(nmodeproduct(nmodeproduct(nn.a{i-1},nn.W1{i-1},1),nn.W2{i-1},2) + nn.b{i-1});
                
            case 'tanh_opt'
                %nn.a{i} = tanh_opt(nn.W1{i-1} * nn.a{i - 1} * nn.W2{i - 1}' + nn.b{i-1});
                nn.a{i} = tanh_opt(nmodeproduct(nmodeproduct(nn.a{i-1},nn.W1{i-1},1),nn.W2{i-1},2) + nn.b{i-1});
			case 'ReLU'
                %nn.a{i} = max(nn.W1{i-1} * nn.a{i - 1} * nn.W2{i - 1}' + nn.b{i-1},0);
                if (nn.fullyconnection== 1 && i+1>l)
                    sizete1 = size(nn.b{i-1});
                    sizete2 = size(nn.W1{i-1} * nn.a{i - 1});
                    nn.a{i} = max(nn.W1{i-1} * nn.a{i - 1} + nn.b{i-1},0);
                else
                    nn.a{i} = max(nmodeproduct(nmodeproduct(nn.a{i-1},nn.W1{i-1},1),nn.W2{i-1},2) + nn.b{i-1},0);
                end
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
        %nn.a{i} = [ones(m,1) nn.a{i}];
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
            if (nn.fullyconnection== 1)
                temp = nn.W1{n-1} * nn.a{n - 1} + nn.b{n-1};
            else
                temp = nmodeproduct(nmodeproduct(nn.a{n-1},nn.W1{n-1},1),nn.W2{n-1},2) + nn.b{n-1};
            end

			nn.a1 = temp(1:dim,:);
			nn.a2 = temp(dim+1:end,:);
            nn.y1= abs(nn.a1)./(abs(nn.a1)+abs(nn.a2)+1e-10);
            nn.y2= abs(nn.a2)./(abs(nn.a1)+abs(nn.a2)+1e-10);
			nn.s1_h = nn.y1.*mix;
			nn.s2_h = nn.y2.*mix;
            nn.a{n} = [nn.s1_h;nn.s2_h];            		
    end
    
    %error and loss
	switch nn.output
	    case {'sigm', 'linear','softmax'}
            nn.e = y - nn.a{n};
		case 'NN_SCSS'
       		nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum( nn.e.^ 2)) ; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) ;
	    case 'NN_SCSS'
		    nn.L = 1/2 * sum(sum( (s1-nn.s1_h).^ 2)) + 1/2 * sum(sum( (s2-nn.s2_h).^ 2)) ;
    end
end
