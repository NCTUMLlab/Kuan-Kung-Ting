function nn = nnapplygrads(nn,batchsize,timestep)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
   

    for i = 1 : (nn.n - 1)
         if(nn.weightPenaltyL2>0)
             dW1 = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
         else
             if(nn.adam>0)
                nn.m1{i} = nn.beta1*nn.m1{i}+(1-nn.beta1)*(nn.dW1{i}/batchsize);
                nn.m2{i} = nn.beta1*nn.m2{i}+(1-nn.beta1)*(nn.dW2{i}/batchsize);
                nn.mb{i} = nn.beta1*nn.mb{i}+(1-nn.beta1)*(nn.db{i}/batchsize);
                nn.v1{i} = nn.beta2*nn.v1{i}+(1-nn.beta2)*((nn.dW1{i}/batchsize).*(nn.dW1{i}/batchsize));
                nn.v2{i} = nn.beta2*nn.v2{i}+(1-nn.beta2)*((nn.dW2{i}/batchsize).*(nn.dW2{i}/batchsize));
                nn.vb{i} = nn.beta2*nn.vb{i}+(1-nn.beta2)*((nn.db{i}/batchsize).*(nn.db{i}/batchsize));
                nn.m_h1 = nn.m1{i}/(1-power(nn.beta1,timestep));
                nn.m_h2 = nn.m2{i}/(1-power(nn.beta1,timestep));
                nn.m_hb = nn.mb{i}/(1-power(nn.beta1,timestep));
                nn.v_h1 = nn.v1{i}/(1-power(nn.beta2,timestep));
                nn.v_h2 = nn.v2{i}/(1-power(nn.beta2,timestep));
                nn.v_hb = nn.vb{i}/(1-power(nn.beta2,timestep));
                dW1 = nn.m_h1./(sqrt(nn.v_h1)+1e-8);
                dW2 = nn.m_h2./(sqrt(nn.v_h2)+1e-8);
                db = nn.m_hb./(sqrt(nn.v_hb)+1e-8);
            else
                 dW1 = nn.dW1{i}/batchsize;
                 dW2 = nn.dW2{i}/batchsize;
                 db = nn.db{i}/batchsize;
            end
             
         end
        
        dW1 = nn.learningRate * dW1;
        dW2 = nn.learningRate *dW2;
        db = nn.learningRate * db;
        
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W1{i} = nn.W1{i} - dW1;
        nn.W2{i} = nn.W2{i} - dW2;
        nn.b{i} = nn.b{i} - db;
        
        
    end
end
