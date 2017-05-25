function [ Z, H, dnorm ] = deepMF_multiview( XX, layers, varargin )
% contain all the stuffs,
% including graph and beta between lost and graph term

% Process optional arguments
pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'TolFun', ...
    'verbose', 'bUpdateZ', 'cache', 'gnd', 'gamma', 'beta', 'graph_k', 'savePath'...
    };



% each view should be initialized also.

numOfView = numel(XX);
num_of_layers = numel(layers);
numOfSample = size(XX{1,1},2);


alpha = ones(numOfView,1).*(1/numOfView);

Z = cell(numOfView, num_of_layers);
H = cell(numOfView, num_of_layers);

dflts  = {0, 0, 1, 1, 100, 1e-5, 1, 1, 0, 0};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ, cache, gnd, gamma, beta, graph_k,savePath] = ...
    internal.stats.parseArgs(pnames,dflts,varargin{:});

A_graph = cell(1,numOfView);
D_graph = cell(1,numOfView);
L_graph = cell(1,numOfView);
options = [];
options.k = graph_k;
options.WeightMode = 'HeatKernel';


for v_ind = 1:numOfView,
    X = XX{v_ind};
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    
    A_graph{v_ind} = constructW(X', options);
    D_graph{v_ind} = diag(sum(constructW(X', options),2));
    L_graph{v_ind} = D_graph{v_ind} - A_graph{v_ind};
    
    
    if  ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = X;
            else
                V = H{v_ind,i_layer-1};
            end
            
            if verbose
                display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
            end
            if ~iscell(z0)
                % For the later layers we use nonlinearities as we go from
                % g(H_{k-1}) to Z*H_k
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', maxiter, ...
                    'bUpdateH', true, 'bUpdateZ', bUpdateZ, 'verbose', verbose, 'save', cache, 'fast', 1);
            else
                display('Using existing Z');
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', 1, ...
                    'bUpdateH', true, 'bUpdateZ', 0, 'z0', z0{i_layer}, 'verbose', verbose, 'save', cache, 'fast', 1);
            end
        end
        
    else
        Z=z0;
        H=h0;
        
        if verbose
            display('Skipping initialization, using provided init matrices...');
        end
    end
    
    
    
    dnorm0(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^gamma, L_graph{v_ind}, beta);
    dnorm(v_ind) = dnorm0(v_ind) + 1;
    
    if verbose
        display(sprintf('#%d error: %f', 0, sum(dnorm0)));
    end
end


%% Error Propagation

if verbose
    display('Finetuning...');
end
H_err = cell(numOfView, num_of_layers);
derror = [];
for iter = 1:maxiter
    Hm_a = 0; Hm_b = 0;
    for v_ind = 1:numOfView,
        X = XX{v_ind};
        
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
        H_err{v_ind,numel(layers)} = H{v_ind,numel(layers)};
        for i_layer = numel(layers)-1:-1:1
            H_err{v_ind,i_layer} = Z{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
        end
        
        for i = 1:numel(layers)
            if bUpdateZ
                try
                    if i == 1
                        Z{v_ind,i} = X  * pinv(H_err{v_ind,1});
                    else
                        Z{v_ind,i} = pinv(D') * X * pinv(H_err{v_ind,i});
                    end
                catch
                    display(sprintf('Convergance error %f. min Z{i}: %f. max %f', norm(Z{v_ind,i}, 'fro'), min(min(Z{v_ind,i})), max(max(Z{v_ind,i}))));
                end
            end
            
            if i == 1
                D = Z{v_ind,1}';
            else
                D = Z{v_ind,i}' * D;
            end
            
            if bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))
                % original one
                A = D * X;
                
                Ap = (abs(A)+A)./2;
                An = (abs(A)-A)./2;
                
                % Hm*A  -> HmA
                HmA =  beta*H{v_ind,i};
                HmAp = (abs(HmA)+HmA)./2;
                HmAn = (abs(HmA)-HmA)./2;
                
                % original noe
                B = D * D';
                Bp = (abs(B)+B)./2;
                Bn = (abs(B)-B)./2;
                
                
                % Hm*D -> HmD
                HmD = beta*H{v_ind,i};
                HmDp = (abs(HmD)+HmD)./2;
                HmDn = (abs(HmD)-HmD)./2;
                
                % update graph part
                
                
                H{v_ind,i} = H{v_ind,i} .* sqrt((Ap + Bn* H{v_ind,i} ) ./ max(An + Bp* H{v_ind,i}, 1e-10));
                
                % set H{v_ind,n_of_layer} = Hm
                % update the last consensus layer
                if i == numel(layers)
                    Hm_a = (alpha(v_ind)^gamma)*(Ap + Bn* H{v_ind,i} + HmAp* A_graph{v_ind} + HmDn* D_graph{v_ind}) + Hm_a;
                    Hm_b = (alpha(v_ind)^gamma)*(max(An + Bp* H{v_ind,i} + HmAn* A_graph{v_ind} + HmDp* D_graph{v_ind}, 1e-10)) + Hm_b;
                end
                
                
            end
        end
        
        
        
        assert(i == numel(layers));
    end
    
    
    for v_ind = 1:numOfView,
        
        X = XX{v_ind};
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
        % update Hm
        H{v_ind,num_of_layers} = H{v_ind,num_of_layers} .* sqrt(Hm_a ./ Hm_b);
        
        % get the error for each view
        dnorm(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^gamma, L_graph{v_ind}, beta);
        
        % the following two lines are used for calculating weight
        tmpNorm = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), 1, L_graph{v_ind}, beta);
        dnorm_w(v_ind) = (gamma*(tmpNorm))^(1/(1-gamma));
    end
    
    % update alpha
    for v_ind = 1:numOfView,
        
        alpha(v_ind) = dnorm_w(v_ind)/sum(dnorm_w);
    end
    
    
    
    
    % finish update Z H and other variables in each view
    % disp result
    
    maxDnorm = sum(dnorm);
    if verbose
        display(sprintf('#%d error: %f', iter, maxDnorm));
        derror(iter) = maxDnorm;
    end
    
    %     assert(dnorm <= dnorm0 + 0.01, ...
    %         sprintf('Rec. error increasing! From %f to %f. (%d)', ...
    %         dnorm0, dnorm, iter) ...
    %     );
    
    if verbose && length(gnd) > 1
        if mod(iter, 1) == 0|| iter ==1
            [acc, nmii, ~ ]= evalResults_multiview(H{numOfView,num_of_layers}, gnd);
            ac = mean(acc);
            ac_std = std(acc);
            nmi = mean(nmii);
            nmi_std = std(nmii);
            
            fprintf(1, 'Clustering accuracy is %.4f, NMI is %.4f\n', ac, nmi);
        end
    end
    
    %             if dnorm0-maxDnorm <= tolfun*max(1,dnorm0)
    %                 if verbose
    %                     display( ...
    %                         sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
    %                             iter, maxDnorm, dnorm0 ...
    %                         ) ...
    %                     );
    %                 end
    %                 break;
    %             end
    
    dnorm0 = maxDnorm;
    
end

end

function error = cost_function(X, Z, H, weight)
error = weight*norm(X - reconstruction(Z, H), 'fro');
end

function error = cost_function_graph(X, Z, H, weight, A, beta)
out = H{numel(H)};
error = weight*(norm(X - reconstruction(Z, H), 'fro') + beta* trace(out*A*out'));
end

function [ out ] = reconstruction( Z, H )

out = H{numel(H)};

for k = numel(H) : -1 : 1;
    out =  Z{k} * out;
end

end
