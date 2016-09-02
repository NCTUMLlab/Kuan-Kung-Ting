function nn = nnbp(nn,mix,s1,s2)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
   
       n = nn.n;
    if (nn.fullyconnection== 1)
       l = n-nn.fullyconnectlayer+1;
    else
       l = n;
    end
    
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
	    case 'NN_SCSS'
		    d1 = ((nn.s1_h-s1)-(nn.s2_h-s2)).*mix.*nn.y2.*(sign(nn.a1)./(abs(nn.a1)+abs(nn.a2)+1e-10));
			d2 = -((nn.s1_h-s1)-(nn.s2_h-s2)).*mix.*nn.y1.*(sign(nn.a2)./(abs(nn.a1)+abs(nn.a2)+1e-10));
			d{n} = [d1;d2];
			%d{n} = d{n}.*(nn.a{n} .* (1 - nn.a{n}));
            
    end   
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
			case 'ReLU'
                d_act = nn.a{i}>=0;			
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1>=l % in this case in d{n} there is not the bias term to be removed 
            if (nn.fullyconnection== 1)
               d{i} = (nn.W1{i}' * d{i + 1}) .* d_act; % Bishop (5.56)
            else
               d{i} = nmodeproduct(nmodeproduct(d{i+1},nn.W1{i}',1),nn.W2{i}',2).* d_act;
            end
        else % in this case in d{i} the bias term has to be removed
            d{i} = nmodeproduct(nmodeproduct(d{i+1},nn.W1{i}',1),nn.W2{i}',2).* d_act;
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1>=l        
              db = d{i+1};
              nn.db{i}  = nn.db{i}+db;
              if (nn.fullyconnection== 1)
                 dW1 = d{i + 1} * nn.a{i}';
                 nn.dW1{i} = nn.dW1{i} + dW1;
              else
                 dW1 =d{i+1}*(nmodeproduct(nn.a{i},nn.W2{i},2))';
                 dW2 =d{i+1}'*(nmodeproduct(nn.a{i},nn.W1{i},1));
                 nn.dW1{i} = nn.dW1{i}+dW1;
                 nn.dW2{i} = nn.dW2{i}+dW2;
              end
        else
            db = d{i+1};
            dW1 =d{i+1}*(nmodeproduct(nn.a{i},nn.W2{i},2))';
            dW2 =d{i+1}'*(nmodeproduct(nn.a{i},nn.W1{i},1));
                        
            nn.db{i}  = nn.db{i}+db;
            nn.dW1{i} = nn.dW1{i}+dW1;
            nn.dW2{i} = nn.dW2{i}+dW2;
        end
    end
end
