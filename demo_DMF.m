% Code for paper "Zhao, et al., Multi-View Clustering via Deep Matrix Factorization, AAAI'17"
% contact handong.zhao@gmail.com if you have any questions
% Nov. 24, 2016


% Courtesy to George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, Bjoern W. Schuller,
% for the codes, from the following paper:
% "George Trigeorgis et al. A Deep Semi-NMF Model for Learning Hidden Representations, ICML'14"


addpath(genpath('.'));
clear;
warning off;
%%

% Yale Dataset -----------------------------
load ./data/yale_mtv.mat;

X{1,1} = NormalizeFea(X{1,1}, 0);
X{1,2} = NormalizeFea(X{1,2}, 0);
X{1,3} = NormalizeFea(X{1,3}, 0);
fea = X;
gnd = gt;
%%
savePath = './result/yale/';

layers = [100 50] ;
graph_k = 5;
gamma = 0.1;
beta = 0.01;

[ Z, H, dnorm ] = deepMF_multiview( fea, layers, 'gnd', gnd,...
    'gamma', gamma, 'beta', beta, 'graph_k', graph_k, 'savePath', savePath);


return