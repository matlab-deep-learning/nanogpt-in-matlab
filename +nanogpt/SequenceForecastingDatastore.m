classdef SequenceForecastingDatastore < matlab.io.Datastore & ...
        matlab.io.datastore.MiniBatchable
    % Datastore for vector sequence forecasting. Extracts a random
    % subsequence from the underlying sequence and outputs it, along with 
    % the same subsequence shifted one step forward in time.

    properties
        MiniBatchSize = 8;
    end

    properties(SetAccess = protected)
        NumObservations = 8;
    end

    properties(Access=private)
        % UnderlyingSequence
        % Main sequence from which the subsequences are extracted. Must
        % have one channel
        UnderlyingSequence

        % SubsequenceLength
        % Length of the subsequences to extract
        SubsequenceLength

        % VocabSize
        % Number of possible unique tokens
        VocabSize
    end

    methods
        function this = SequenceForecastingDatastore(sequence, subseqlength, vocabSize)
            this.UnderlyingSequence = sequence;
            this.SubsequenceLength = subseqlength;
            this.VocabSize = vocabSize;
        end

        function [data,info] = read(this)
            totalNumSteps = length(this.UnderlyingSequence);
            predictors = zeros(this.MiniBatchSize, this.SubsequenceLength);
            responses = zeros(this.MiniBatchSize, this.SubsequenceLength);

            for ii = 1:this.MiniBatchSize
                startIdx = randi(totalNumSteps - this.SubsequenceLength);
                predictors(ii,:) = this.UnderlyingSequence(startIdx:startIdx+this.SubsequenceLength-1);
                responses(ii,:) = this.UnderlyingSequence(startIdx+1:startIdx+this.SubsequenceLength);
            end

            responses = categorical(responses, 1:this.VocabSize);

            data = table(predictors,responses);
            info = struct();
        end

        function tf = hasdata(~)
            % Data is extracted randomly, so we always return true
            tf = true;
        end

        function reset(~)
            % Datasore has no state, so this is a no-op
        end

        function p = progress(~)
            % Data never runs out, so return 0
            p = 0;
        end
    end
end