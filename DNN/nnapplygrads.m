function nn = nnapplygrads(nn,timestep)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            if(nn.adam>0)
                nn.m{i} = nn.beta1*nn.m{i}+(1-nn.beta1)*nn.dW{i};                
                nn.v{i} = nn.beta2*nn.v{i}+(1-nn.beta2)*(nn.dW{i}.*nn.dW{i});
                nn.m_h = nn.m{i}/(1-power(nn.beta1,timestep));                
                nn.v_h = nn.v{i}/(1-power(nn.beta2,timestep));
                dW = nn.m_h./(sqrt(nn.v_h)+1e-8);
 
            else
                 dW = nn.dW{i};
            end
        end
               
        dW = nn.learningRate*dW;
        
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end
end
