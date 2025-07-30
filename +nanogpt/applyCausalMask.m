function out = applyCausalMask(in)
% Applies a causal mask to an input CBT dlarray

maskSize = size(in,[1 3]);
mask = triu(ones(maskSize));
mask = permute(mask, [1 3 2]);

% Set masked elements to -Inf(ish)
out = in .* mask - (1e10) .* (~mask);
end