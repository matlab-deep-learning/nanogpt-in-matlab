function out = batchmtimes(X,transposeX,Y,transposeY)
% Paged matrix multiplication of a CBT dlarray, using the batch dimension
% as the page dimension
% 
% X - C1*B*T1 dlarray
% Y - C2*B*T2 dlarray
%
% Same transpose syntax as pagemtimes

% We want the batch dimensions to be the pages. Strip the dimension labels
% and permute to TCB
X = stripdims(X);
Y = stripdims(Y);
X = permute(X, [3 1 2]);
Y = permute(Y, [3 1 2]); 

out = pagemtimes(X, transposeX, Y, transposeY);

% Permute back to CBT
out = permute(out, [2 3 1]);
end