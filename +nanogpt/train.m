function [net, validationLoss] = train(net, mbqTrain, mbqValidation, opts)
arguments
    net
    mbqTrain
    mbqValidation
    opts.NumIterations = 1000;
    opts.LearnRate = 1e-3;
    opts.ValidationFrequency = 100;
    opts.NumValidationIters = 20;
end

averageGrad = [];
averageSqGrad = [];

monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLoss", "ValidationLoss"], ...
    XLabel="Iteration");
groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);
iteration = 0;
while iteration < opts.NumIterations && ~monitor.Stop
    iteration = iteration + 1;
    [X,T] = mbqTrain.next();
    [loss, gradients] = dlfeval(@nanogpt.modelLoss, net, X, T);
    [net,averageGrad,averageSqGrad] = adamupdate(net,gradients, ...
            averageGrad,averageSqGrad,iteration,opts.LearnRate);
    recordMetrics(monitor,iteration,TrainingLoss=loss);

    % Every validation frequency, validate on numValidationIters minibatches from the validation set
    if mod(iteration, opts.ValidationFrequency) == 0 || iteration == 1
        validationLoss = 0;
        for ii = 1:opts.NumValidationIters
            [XVal, TVal] = mbqValidation.next();
            validationLoss = validationLoss + nanogpt.modelLoss(net, XVal, TVal)/opts.NumValidationIters;
        end
        recordMetrics(monitor,iteration,ValidationLoss=validationLoss);
    end
end
end