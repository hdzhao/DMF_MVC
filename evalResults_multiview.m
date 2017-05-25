function [ ACC, nmii, H ] = evalResults_multiview( H, gnd,varargin )

if length(varargin)>0
    kk = varargin{1};
end

nClass = length(unique(gnd));

if iscell(H)
    H = H{:};
end

for iLoop = 1:10,
    
    options = [];
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    A = constructW(H', options);
    gt = gnd; clusNum = nClass;
    
    C = SpectralClustering(A,clusNum);
    
    [A nmii(iLoop) avgent] = compute_nmi(gt,C);
    [F,P,R] = compute_f(gt,C);
    [AR,RI,MI,HI]=RandIndex(gt,C);
    C = bestMap(gt,C);
    ACC(iLoop) = length(find(gt == C))/length(gt);
    
    
    
end


end

