function [v1, v2, v3]=VaR(data, alp)

% -----------
% Purpose : Value-at-Risk estimation 
% -----------

% -- Input --
% data: a column data vector 
% alp:  significance level of 
% -----------


% -- Output --
% v1: VaR as the alp-quantile by its formal definition 
%                (alp quasi-inverse of the CDF of data)
% v2: Conservative VaR (bigger loss estimation than v1)
% v3: VaR as the alp-quantile by "liner interpolation"
%                (not the formal definition of quasi-inverse of CDF)
% -----------

n = numel(data); 
d = sort(data); 

an = alp * n;

a1 = floor(an); 
a2 = ceil(an); 

v1 = d(a2); 
v2 = d(a1);
v3 = (a2-an)*v2 + (an-a1)*v1;

% VaR is a positive number representing the loss
v1=-v1;
v2=-v2;
v3=-v3;
