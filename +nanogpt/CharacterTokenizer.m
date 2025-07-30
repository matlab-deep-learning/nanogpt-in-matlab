classdef CharacterTokenizer < handle
    % CharacterTokenizer  Converts between characters and tokens

    properties(SetAccess=private)
        % VocabSize  (numeric)
        % Number of unique characters
        VocabSize
    end

    properties(Access=private)
        % EncodingDict  (dictionary)
        % Dictionary between characters and tokens
        EncodingDict

        % DecodingDict  (dictionary)
        % Dictionary between tokens and characters
        DecodingDict
    end

    methods
        function this = CharacterTokenizer(document)
            % Expects document as a char array
            uniqueChars = unique(document);
            this.VocabSize = numel(uniqueChars);
            % The tokens are integers, one for each unique character in the
            % document, in alphabetical order.
            tokens = 1:numel(uniqueChars);
            this.EncodingDict = dictionary(uniqueChars', tokens');
            this.DecodingDict = dictionary(tokens', uniqueChars');
        end

        function tokens = char2tok(this, chars)
            % Converts a char array to a vector of tokens
            tokens = this.EncodingDict(chars(:));
            % Reshape
            tokens = reshape(tokens, size(chars));
        end

        function chars = tok2char(this, tokens)
            % Converts an array of integer tokens to a char array
            chars = char(this.DecodingDict(tokens(:)));
            % Reshape
            chars = reshape(chars, size(tokens));
        end

        function chars = oneHotTok2Char(this, tokens)
            % Converts a one-hot encoded array of tokens to a char array
            tokens = onehotdecode(tokens, 1:this.VocabSize, 3);
            chars = char(this.DecodingDict(tokens(:)));
            % Reshape
            chars = reshape(chars, size(tokens));
        end
    end
end