function newChars = generateText(net, context, numChars, tokenizer, maxContextLength, opts)
arguments
    net
    context
    numChars
    tokenizer
    maxContextLength
    opts.Verbose = false
end
% Tokenize the context
if ~isnumeric(context)
    context = tokenizer.char2tok(context);
    context = dlarray(context, 'BTC');
end
if opts.Verbose
    fprintf(tokenizer.tok2char(context));
end
% Trim the context to fit the max context length
if size(context, 3) > maxContextLength
    context = context(:, :, end-maxContextLength:end);
end

currentContext = context;
numBatches = size(context, 2);
if ~opts.Verbose
    newChars = repmat('a', [numBatches numChars]);
end
ii = 0;
while ii < numChars
    ii = ii+1;
    prediction = net.predict(currentContext); % CBT dlarray

    lastPrediction = prediction(:,:,end); % 1BT dlarray
    predictedToken = sampleFromLogits(extractdata(lastPrediction)); % 1B1 categorical
    if size(currentContext, 3) == maxContextLength
        currentContext = cat(3, currentContext(:,:,2:end), predictedToken);
    else
        currentContext = cat(3, currentContext, predictedToken);
    end

    nextToken = squeeze(predictedToken); %B1 double
    nextChar = tokenizer.tok2char(nextToken);

    if opts.Verbose
        fprintf("%s", nextChar);
        drawnow
    else
        newChars(:,ii) = nextChar;
    end
end
if opts.Verbose
    fprintf("\n");
end
end

function token = sampleFromLogits(logits)
% Given a vector of probabilities, sample a token with the probability
% weight given by the elements of that vector

population = 1:size(logits,1);
numBatches = size(logits,2);
token = zeros(1,numBatches);
for ii = 1:numBatches
    token(:,ii) = randsample(population, 1, true, double(logits(:,ii)));
end
end